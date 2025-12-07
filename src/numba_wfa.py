import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import time
import itertools
from datetime import datetime, timedelta
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller

from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed

API_KEY_ID = os.getenv("ALPACA_KEY_ID")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

if not API_KEY_ID or not API_SECRET_KEY:
    raise ValueError("API keys not found. Check your .env file and .gitignore.")

ASSET_A = os.getenv("ASSET_A", "V")
ASSET_B = os.getenv("ASSET_B", "MA")

#caching so i dont pay the troll toll every time i backtest
MASTER_START_DATE = datetime(2021, 1, 1)
MASTER_END_DATE = datetime(2025, 11, 1)

DATA_CACHE_DIR = "market_data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

start_date_env = os.getenv("START_DATE", "")
if start_date_env:
    try:
        START_DATE = datetime.strptime(start_date_env, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid START_DATE format. Use YYYY-MM-DD.")
else:
    START_DATE = datetime.now() - timedelta(days=4 * 365 + 1)

END_DATE = START_DATE + timedelta(days=4 * 365)

time_config = os.getenv("TIME_CONFIG", "15m")
if time_config.endswith("m"):
    SELECTED_TIMEFRAME = TimeFrame.Minute
    SELECTED_TIMEFRAME_VALUE = int(time_config[:-1])
elif time_config.endswith("h"):
    SELECTED_TIMEFRAME = TimeFrame.Hour
    SELECTED_TIMEFRAME_VALUE = int(time_config[:-1])
elif time_config.endswith("d"):
    SELECTED_TIMEFRAME = TimeFrame.Day
    SELECTED_TIMEFRAME_VALUE = int(time_config[:-1])
else:
    raise ValueError(f"Unsupported TIME_CONFIG format: {time_config}")

# costs
slippage = 0.01 # per share in cnets
txfee = 0.0001 # this is whole number, *100 for percentage

ROLLING_SHIFT_DAYS = 15 # 15 day out of sample, the config for in-sample is down there somewhere
BASE_RTH_MINUTES = 390
TRADING_DAYS_PER_YEAR = 252

if SELECTED_TIMEFRAME.unit == TimeFrameUnit.Minute:
    TIME_UNIT_IN_MINUTES = SELECTED_TIMEFRAME_VALUE
elif SELECTED_TIMEFRAME.unit == TimeFrameUnit.Hour:
    TIME_UNIT_IN_MINUTES = SELECTED_TIMEFRAME_VALUE * 60
elif SELECTED_TIMEFRAME.unit == TimeFrameUnit.Day:
    TIME_UNIT_IN_MINUTES = BASE_RTH_MINUTES
else:
    raise ValueError(f"Unsupported TimeFrame unit: {SELECTED_TIMEFRAME.unit}")

BARS_PER_TRADING_DAY = int(BASE_RTH_MINUTES / TIME_UNIT_IN_MINUTES)
if BARS_PER_TRADING_DAY == 0:
    BARS_PER_TRADING_DAY = 1

BARS_PER_YEAR = BARS_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR

MIN_STAT_SAMPLES = 60
CALC_BARS = int(15 * BARS_PER_TRADING_DAY)
ROLLING_WINDOW_BARS = max(CALC_BARS, MIN_STAT_SAMPLES)

OPTIMIZATION_WINDOW_DAYS = 60 # in-sample days, putting 4:1 ratio as per some literature
OPTIMIZATION_WINDOW_BARS = int(OPTIMIZATION_WINDOW_DAYS * BARS_PER_TRADING_DAY)

ROLLING_WINDOW_BARS = max(ROLLING_WINDOW_BARS, 5)
OPTIMIZATION_WINDOW_BARS = max(OPTIMIZATION_WINDOW_BARS, 50)

print(f"TimeFrame Selected: {SELECTED_TIMEFRAME.value}, Value: {SELECTED_TIMEFRAME_VALUE}")
print(f"Bars per Trading Day: {BARS_PER_TRADING_DAY}")
print(f"Annualization Factor (Bars/Year): {BARS_PER_YEAR}")
print(f"Statistical Lookback Bars: {ROLLING_WINDOW_BARS}")
print(f"Optimization Lookback Bars: {OPTIMIZATION_WINDOW_BARS}")

cptl = 10000.0
abs_stop = 250.0

Z_ENTRY_GRID = [1.7, 1.9, 2.1, 2.3, 2.5]
Z_EXIT_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
Z_STOP_LOSS_GRID = [3.0, 3.3, 3.6, 3.9, 4.2, 4.5]

ADF_P_VALUE_THRESHOLD = float(os.getenv("ADF_P_VALUE_THRESHOLD", 0.20))
hurstMax = 0.75 # google said <0.5 is mean-reverting, but empirically i found 0.75-0.8 to be better hurst limits which let trades in without suffocating the whole thing

# gonna log it to excel for persistence. quite useless now that im using numba so each run on each wfa is like 20s
OUTPUT_DIR = "wfa_outputs_numba"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"oos_equity_curve_{ASSET_A}_{ASSET_B}_{time_config}_{timestamp}_{ADF_P_VALUE_THRESHOLD}.csv")

try:
    data_client = StockHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
except Exception as e:
    print(f"!!! error initializing StockHistoricalDataClient: {e} !!!")
    raise

def get_alpaca_data(symbol, req_start, req_end, timeframe, timeframe_value):
    tf_str = f"{timeframe_value}{timeframe.unit.value}"
    master_start_str = MASTER_START_DATE.strftime("%Y%m%d")
    master_end_str = MASTER_END_DATE.strftime("%Y%m%d")
    filename = f"{symbol}_{tf_str}_MASTER_{master_start_str}_{master_end_str}.pkl"
    file_path = os.path.join(DATA_CACHE_DIR, filename)
    full_data = None

    if os.path.exists(file_path):
        print(f"found cache, loading data for {symbol}...")
        try:
            full_data = pd.read_pickle(file_path)
        except Exception as e:
            print(f"!!! cache corrupted, re-downloading master. Error: {e} !!!")

    if full_data is None:
        print(f"Downloading history for {symbol} ({master_start_str}-{master_end_str})...")
        if timeframe.unit == TimeFrameUnit.Minute and timeframe_value != 1:
            tf = TimeFrame(timeframe_value, TimeFrameUnit.Minute)
        elif timeframe.unit == TimeFrameUnit.Hour and timeframe_value != 1:
            tf = TimeFrame(timeframe_value, TimeFrameUnit.Hour)
        else:
            tf = timeframe

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=MASTER_START_DATE,
            end=MASTER_END_DATE,
            feed=DataFeed.SIP,
            adjustment="all"
        )

        for attempt in range(3):
            try:
                bars = data_client.get_stock_bars(request_params).df
                close_prices = bars.loc[(symbol, slice(None)), 'close'].rename(f'Close_{symbol}')
                full_data = close_prices.droplevel('symbol')

                # save the file so i dont have to keep downloading, used to take +2 mins per run (and i did A LOT of runs downloading data everytime)
                print(f"Saving history in a cache at {file_path}")
                full_data.to_pickle(file_path)
                break
            except Exception as e:
                if "401" in str(e): raise e
                print(f"!!! Attempt {attempt + 1} failed: {e} !!!")
                time.sleep(2)

        if full_data is None:
            raise Exception(f"Failed to download master data for {symbol}")

    if full_data.index.tz is None:
        full_data = full_data.tz_localize('UTC')

    # handles the problems in the evnet req_start is timezone naive
    slice_start = pd.Timestamp(req_start).tz_localize("UTC") if pd.Timestamp(req_start).tz is None else req_start
    slice_end = pd.Timestamp(req_end).tz_localize("UTC") if pd.Timestamp(req_end).tz is None else req_end

    sliced_data = full_data.loc[slice_start:slice_end]
    print(f"Sliced {len(sliced_data)} bars from Master ({len(full_data)} total)")
    return sliced_data


def prepare_data(asset_a, asset_b, start_date, end_date, timeframe, timeframe_value):
    print(f"preparing Data for {asset_a} & {asset_b}")
    df_a = get_alpaca_data(asset_a, start_date, end_date, timeframe, timeframe_value)
    df_b = get_alpaca_data(asset_b, start_date, end_date, timeframe, timeframe_value)

    print("merging and aligning data...")
    df = pd.concat([df_a, df_b], axis=1, join='inner')

    df.columns = [f'Close_{asset_a}', f'Close_{asset_b}']

    df['Log_A'] = np.log(df[f'Close_{asset_a}'])
    df['Log_B'] = np.log(df[f'Close_{asset_b}'])

    print(f"data Prepared. {len(df)} aligned bars ready.")
    return df

from numba import float64, njit

@njit
def calc_kalman(y, x, delta=1e-4, ve=1e-3): # delta and ve are arbitrarily selected but this seems ok so far
    # i literally have no words to explain how i came to these numbers, but source: trust me bro
    # *kinda* learn the math from this www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
    n = len(y)
    beta = np.zeros(n)
    state_mean = 0.0
    state_cov = 0.1

    for t in range(n):
        state_cov = state_cov + delta
        obs_val = y[t]
        obs_mat = x[t]
        pred_obs = obs_mat * state_mean
        residual = obs_val - pred_obs
        variance = (obs_mat * state_cov * obs_mat) + ve
        kalman_gain = (state_cov * obs_mat) / variance
        state_mean = state_mean + kalman_gain * residual
        state_cov = (1 - kalman_gain * obs_mat) * state_cov
        beta[t] = state_mean
    return beta

@njit
def calc_hurst(series, window=100):
    n = len(series)
    hurst = np.full(n, 0.5) # defaults to 0.5 but that will literally block everything
    min_window = max(window, 30)

    for t in range(min_window, n):
        ts = series[t - window:t]
        mean_ts = np.mean(ts)
        centered = ts - mean_ts
        cumulative = np.cumsum(centered)
        R = np.max(cumulative) - np.min(cumulative)
        S = np.std(ts)
        if S == 0:
            hurst[t] = 0.5
        else:
            hurst[t] = np.log(R / S) / np.log(len(ts))
    return hurst


def calc_spread_z(data, lookback):
    Y = data['Log_A'].values
    X = data['Log_B'].values
    data['Beta'] = calc_kalman(Y, X)

    rolling_mean_Y = data['Log_A'].rolling(window=lookback).mean()
    rolling_mean_X = data['Log_B'].rolling(window=lookback).mean()

    data['Alpha'] = rolling_mean_Y - data['Beta'] * rolling_mean_X

    data['Spread'] = data['Log_A'] - (data['Alpha'] + data['Beta'] * data['Log_B'])

    rolling_mean_spread = data['Spread'].shift(1).rolling(window=lookback).mean()
    rolling_std_spread = data['Spread'].shift(1).rolling(window=lookback).std()

    data['Spread_Std'] = rolling_std_spread

    data['Z_Score'] = (data['Spread'] - rolling_mean_spread) / (rolling_std_spread + 1e-9)

    hurst_win = 100 # a number picked as a balance btwn statistical accuracy and responsiveness, otherwise unjustifiable.
    data['Hurst'] = calc_hurst(data['Spread'].values, window=hurst_win)

    data.dropna(subset=['Beta', 'Alpha', 'Z_Score', 'Hurst', 'Spread_Std'], inplace=True)

    return data

def chk_adf(is_data, asset_a, asset_b):
    #using OLS ONLY for kalman since kalman ensures the beta error is minimal and hence adf would be deceptively low

    if is_data.empty:
        return 1.0, 0.0
    Y = is_data['Log_A']
    X = is_data['Log_B']
    X = add_constant(X)

    model = OLS(Y, X).fit()
    residuals = model.resid

    adf_result = adfuller(residuals)
    p_value = adf_result[1]

    ols_beta = model.params['Log_B']

    return p_value, ols_beta

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
                 float64, float64, float64, float64, float64, float64, float64, float64))
def numba_bt(close_a, close_b, z_score, beta, hurst, volatility, z_entry, z_exit, z_stop_loss, cptl, txfee,
                        slippage, abs_stop, hurstMax):
# honest to god, numba is magic, idk what is done, the way it was transplanted to cpp is also esentially black magic to me
# a 1600s peasant would instantaneously combust if he was shown numba
    n_bars = len(close_a)
    pnl_array = np.zeros(n_bars, dtype=np.float64)
    position = 0.0

    entry_price_a = 0.0
    entry_price_b = 0.0
    entry_shares_a = 0.0
    entry_shares_b = 0.0
    n_A = 0.0
    n_B = 0.0

    for i in range(n_bars - 1):

        current_z = z_score[i]
        current_hurst = hurst[i]
        prev_position = position

        next_price_a = close_a[i + 1]
        next_price_b = close_b[i + 1]
        current_beta = beta[i]

        is_final_bar = (i == n_bars - 2)

        if prev_position != 0:
            is_mean_reversion_exit = np.abs(current_z) <= z_exit
            is_stop_loss_exit = np.abs(current_z) >= z_stop_loss
            is_dollar_stop_loss_exit = False
            if entry_price_a > 0 and entry_price_b > 0:
                if prev_position == 1:  #long
                    pnl_a = (next_price_a - entry_price_a) * entry_shares_a
                    pnl_b = (entry_price_b - next_price_b) * entry_shares_b
                else:  #short
                    pnl_a = (entry_price_a - next_price_a) * entry_shares_a
                    pnl_b = (next_price_b - entry_price_b) * entry_shares_b
                total_gross_pnl = pnl_a + pnl_b
                if total_gross_pnl <= -abs_stop:
                    is_dollar_stop_loss_exit = True

            if is_mean_reversion_exit or is_stop_loss_exit or is_dollar_stop_loss_exit or is_final_bar:

                if entry_price_a > 0 and entry_price_b > 0:
                    n_A = entry_shares_a
                    n_B = entry_shares_b

                    if prev_position == 1:
                        pnl_a = (next_price_a - entry_price_a) * n_A
                        pnl_b = (entry_price_b - next_price_b) * n_B
                    else:
                        pnl_a = (entry_price_a - next_price_a) * n_A
                        pnl_b = (next_price_b - entry_price_b) * n_B

                    gross_pnl = pnl_a + pnl_b

                    entry_value = entry_price_a * n_A + entry_price_b * n_B
                    exit_value = next_price_a * n_A + next_price_b * n_B
                    cost_fee_percent = txfee * (entry_value + exit_value)
                    cost_slippage = slippage * (n_A + n_B) * 2.0
                    total_cost_trade = cost_fee_percent + cost_slippage

                    total_pnl_trade = gross_pnl - total_cost_trade

                    pnl_array[i + 1] = total_pnl_trade
                    position = 0.0

                    entry_price_a = 0.0
                    entry_price_b = 0.0
                    entry_shares_a = 0.0
                    entry_shares_b = 0.0

        elif prev_position == 0 and not is_final_bar:
            if np.abs(current_z) >= z_entry and current_hurst < hurstMax:

                beta_val = current_beta

                if np.isnan(beta_val):
                    continue

                current_vol = volatility[i]
                vol_scale = 1.0
                if current_vol > 0:
                    vol_scale = 0.15 / current_vol
                    if vol_scale > 1.0: vol_scale = 1.0

                adj_capital = cptl * vol_scale

                V_A_entry = adj_capital / (1.0 + np.abs(beta_val))
                V_B_entry = adj_capital - V_A_entry

                n_A = np.floor(V_A_entry / next_price_a)
                n_B = np.floor(V_B_entry / next_price_b)

                entry_price_a = next_price_a
                entry_price_b = next_price_b
                entry_shares_a = n_A
                entry_shares_b = n_B

                if current_z < -z_entry:
                    position = 1.0
                else:
                    position = -1.0

    return pnl_array


def bt_strat(data, z_entry, z_exit, z_stop_loss, cptl, txfee, slippage, asset_a, asset_b, abs_stop):
    close_a = data[f'Close_{asset_a}'].values
    close_b = data[f'Close_{asset_b}'].values
    z_score = data['Z_Score'].values
    beta = data['Beta'].values
    hurst = data['Hurst'].values
    vol = data['Spread_Std'].values

    temp_data = data.copy()

    numba_array = numba_bt(
        close_a, close_b, z_score, beta, hurst, vol,
        z_entry, z_exit, z_stop_loss, cptl, txfee,
        slippage, abs_stop, hurstMax
    )

    if len(numba_array) != len(temp_data):
        numba_array = np.zeros(len(temp_data), dtype=np.float64)

    temp_data['Pnl'] = numba_array

    if len(temp_data) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, [], pd.DataFrame({'Time': [], 'PnL': []})

    temp_data['Cumulative_Pnl'] = temp_data['Pnl'].cumsum()
    temp_data['Returns'] = temp_data['Pnl'] / cptl

    total_pnl = float(temp_data['Cumulative_Pnl'].iloc[-1])
    peak = temp_data['Cumulative_Pnl'].cummax()
    drawdown = temp_data['Cumulative_Pnl'] - peak
    max_dollar_drawdown = float(abs(drawdown.min()))

    returns = temp_data['Returns']
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=0)
    if std_ret != 0:
        annualized_returns = mean_ret * BARS_PER_YEAR
        annualized_std = std_ret * np.sqrt(BARS_PER_YEAR)
        sharpe_ratio = annualized_returns / annualized_std
    else:
        sharpe_ratio = 0.0

    if len(temp_data.index) >= 2:
        time_span = (temp_data.index[-1] - temp_data.index[0]).total_seconds() / (60.0 * 60.0 * 24.0)
        years = max(time_span / 365.0, 1.0 / 252.0)
    else:
        years = 1.0 / 252.0

    final_equity = cptl + total_pnl
    if final_equity > 0:
        cagr = (final_equity / cptl) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    max_pct_drawdown = (max_dollar_drawdown / cptl) if cptl != 0 else 0.0
    calmar_ratio = cagr / max_pct_drawdown if max_pct_drawdown != 0 else 0.0

    trade_count = (numba_array != 0).sum()
    trade_log = []
    all_pnls = pd.DataFrame({'Time': temp_data.index, 'PnL': numba_array})
    all_pnls = all_pnls[all_pnls['PnL'] != 0].reset_index(drop=True)

    return total_pnl, sharpe_ratio, calmar_ratio, max_dollar_drawdown, trade_count, trade_log, all_pnls


def run_opt(data, cptl, txfee, slippage, asset_a, asset_b, abs_stop, hurstMax):
    param_combinations = list(itertools.product(Z_ENTRY_GRID, Z_EXIT_GRID, Z_STOP_LOSS_GRID))
    best_sharpe = -np.inf
    best_params = None
    best_calmar = 0.0

    close_a = data[f'Close_{asset_a}'].values
    close_b = data[f'Close_{asset_b}'].values
    z_score = data['Z_Score'].values
    beta = data['Beta'].values
    hurst = data['Hurst'].values
    vol = data['Spread_Std'].values

    for z_entry, z_exit, z_stop_loss in param_combinations:
        if not (z_exit < z_entry < z_stop_loss):
            continue

        pnl_array = numba_bt(
            close_a, close_b, z_score, beta, hurst, vol,
            z_entry, z_exit, z_stop_loss, cptl, txfee, slippage,
            abs_stop, hurstMax
        )

        if np.sum(np.abs(pnl_array)) == 0:
            sharpe = 0.0
        else:
            returns = pd.Series(pnl_array) / cptl
            annualized_returns = returns.mean() * BARS_PER_YEAR
            annualized_std = returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR)
            sharpe = annualized_returns / annualized_std if annualized_std != 0 else 0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (z_entry, z_exit, z_stop_loss)

    best_calmar = 0.0
    if best_params:
        z_entry_opt, z_exit_opt, z_stop_loss_opt = best_params
        total_pnl, _, best_calmar, _, _, _, _ = bt_strat(
            data,
            z_entry_opt, z_exit_opt, z_stop_loss_opt,
            cptl, txfee, slippage, asset_a, asset_b, abs_stop
        )

    return best_params, best_sharpe, best_calmar

def rolling_bt(processed_full_data, rolling_window_bars, optimization_window_bars, shift_days, cptl, txfee, slippage, asset_a, asset_b):
    print(f"Starting WFA: IS window size: {optimization_window_bars} bars, Shift: {shift_days} days")
    print(f"Rolling window: {rolling_window_bars} bars")
    print(f"ADF max: {ADF_P_VALUE_THRESHOLD}")

    shift_bars = shift_days * BARS_PER_TRADING_DAY
    required_start_data = optimization_window_bars + shift_bars

    if optimization_window_bars < rolling_window_bars + 1:
        raise ValueError("Optimization Window must be larger than Statistical Rolling Window.")

    if len(processed_full_data) < required_start_data:
        raise ValueError(f"Raw Dataframe too short. Need {required_start_data} bars.")

    all_oos_pnls = []
    total_oos_trades = 0
    window_count = 0
    i = optimization_window_bars

    while i < len(processed_full_data):
        window_count += 1
        is_start_index = max(0, i - optimization_window_bars)
        is_end_index = i

        raw_is_data = processed_full_data.iloc[is_start_index:is_end_index].copy()
        is_data_for_opt = calc_spread_z(raw_is_data, ROLLING_WINDOW_BARS)

        oos_end_index = min(len(processed_full_data), i + shift_bars)
        oos_test_data = processed_full_data.iloc[is_end_index:oos_end_index].copy()

        if len(oos_test_data) < 2:
            print(f"Window {window_count}: not enough data for oos, stopping wfa")
            break

        print(f"\n--- Window {window_count} Boundary Check ---")
        print(f"IS Period RTH Bars: {len(is_data_for_opt)}. Start: {is_data_for_opt.index[0]} | End: {is_data_for_opt.index[-1]}")
        print(f"OOS Start Bar Index: {is_end_index} | Time: {oos_test_data.index[0]}")

        if is_data_for_opt.empty:
            print("Skipping window: IS data empty after calculations.")
            i += shift_bars
            continue

        adf_p_value, ols_beta = chk_adf(is_data_for_opt, asset_a, asset_b)
        print(
            f"ADF Test Result: p-value={adf_p_value:.4f} (Required < {ADF_P_VALUE_THRESHOLD:.2f}) | OLS Beta: {ols_beta:.4f}")

        if adf_p_value >= ADF_P_VALUE_THRESHOLD:
            print(f"ADF filter on: spread not stationary. Skipping.")
            i += shift_bars
            continue

        print("Filter off: spread is stationary. Proceeding with optimization.")

        opt_start_time = time.time()
        best_params, best_sharpe, best_calmar = run_opt(
        is_data_for_opt, cptl, txfee, slippage, asset_a, asset_b, abs_stop, hurstMax)
        opt_end_time = time.time()
        print(f"IS Optimization Time: {opt_end_time - opt_start_time:.2f} seconds.")

        if not best_params:
            print(f"IS Period: No valid parameters found. Skipping.")
            i += shift_bars
            continue

        z_entry_opt, z_exit_opt, z_stop_loss_opt = best_params
        print(f"Optimal Params (Sharpe={best_sharpe:.4f}, Calmar={best_calmar:.4f}): Z_E={z_entry_opt:.1f}, Z_X={z_exit_opt:.1f}, Z_SL={z_stop_loss_opt:.1f}")

        combined_df = pd.concat([raw_is_data, oos_test_data])
        combined_df = calc_spread_z(combined_df, ROLLING_WINDOW_BARS)
        oos_test_data_with_metrics = combined_df.loc[oos_test_data.index].copy()

        total_oos_pnl, oos_sharpe, oos_calmar, oos_mdd, oos_trades, _, oos_pnls_df = bt_strat(
            oos_test_data_with_metrics, z_entry_opt, z_exit_opt, z_stop_loss_opt,
            cptl, txfee, slippage, asset_a, asset_b, abs_stop)

        total_oos_trades += oos_trades
        if not oos_pnls_df.empty:
            all_oos_pnls.append(oos_pnls_df)

        try:
            if all_oos_pnls:
                current_oos_df = pd.concat(all_oos_pnls).sort_values(by='Time').set_index('Time')
                current_oos_df['Cumulative_PnL'] = current_oos_df['PnL'].cumsum()
                current_oos_df.to_csv(OUTPUT_FILE)
                print(f"Persistence: Saved {len(current_oos_df)} PnL points to {OUTPUT_FILE}")
        except Exception as e:
            print(f"!!! Warning: Failed to save to CSV. Error: {e} !!!")

        print(f"OOS Period ({oos_test_data.index[0]} to {oos_test_data.index[-1]}):")
        print(f"PnL: ${total_oos_pnl:,.2f} | Sharpe: {oos_sharpe:.4f} | Calmar: {oos_calmar:.4f} | Trades: {oos_trades}")
        i += shift_bars

    if not all_oos_pnls:
        print("No trades generated in any OOS window.")
        return {
            'Total_PnL': 0.0,
            'Sharpe_Ratio': 0.0,
            'Calmar_Ratio': 0.0,
            'Max_Drawdown': 0.0,
            'Total_Trades': 0,
            'Total_Windows': window_count
        }

    final_pnls_df = pd.concat(all_oos_pnls).sort_values(by='Time').set_index('Time')
    final_pnls_df['Cumulative_Pnl'] = final_pnls_df['PnL'].cumsum()

    if final_pnls_df['Cumulative_Pnl'].empty:
        return {
            'Total_PnL': 0.0,
            'Sharpe_Ratio': 0.0,
            'Calmar_Ratio': 0.0,
            'Max_Drawdown': 0.0,
            'Total_Trades': 0,
            'Total_Windows': window_count
        }

    final_equity = cptl + final_pnls_df['Cumulative_Pnl'].iloc[-1]
    time_span = final_pnls_df.index[-1] - final_pnls_df.index[0]
    total_rth_minutes = time_span.total_seconds() / 60.0
    TRADING_DAYS_IN_PERIOD = total_rth_minutes / 390.0
    YEARS_TESTED = TRADING_DAYS_IN_PERIOD / TRADING_DAYS_PER_YEAR

    if YEARS_TESTED < (1 / TRADING_DAYS_PER_YEAR): YEARS_TESTED = 1 / TRADING_DAYS_PER_YEAR

    if final_equity > 0:
        CAGR = (final_equity / cptl) ** (1.0 / YEARS_TESTED) - 1.0
    else:
        CAGR = -1.0

    overall_peak = final_pnls_df['Cumulative_Pnl'].expanding(min_periods=1).max()
    overall_drawdown = final_pnls_df['Cumulative_Pnl'] - overall_peak
    overall_max_dollar_drawdown = abs(overall_drawdown.min())
    overall_max_percentage_drawdown = overall_max_dollar_drawdown / cptl if cptl != 0 else 0
    overall_calmar_ratio = CAGR / overall_max_percentage_drawdown if overall_max_percentage_drawdown != 0 else 0

    all_trade_pnls = pd.concat(all_oos_pnls)['PnL'].values
    trade_pnls = all_trade_pnls[all_trade_pnls != 0]

    if len(trade_pnls) == 0:
        overall_sharpe_ratio = 0.0
    else:
        mean_trade = trade_pnls.mean()
        std_trade = trade_pnls.std(ddof=0)
        total_years = YEARS_TESTED
        trades_per_year = len(trade_pnls) / total_years
        if std_trade != 0:
            overall_sharpe_ratio = (mean_trade / std_trade) * np.sqrt(trades_per_year)
        else:
            overall_sharpe_ratio = 0.0

    overall_pnl = final_pnls_df['Cumulative_Pnl'].iloc[-1]
    return {
        'Total_PnL': overall_pnl,
        'Sharpe_Ratio': overall_sharpe_ratio,
        'Calmar_Ratio': overall_calmar_ratio,
        'Max_Drawdown': overall_max_dollar_drawdown,
        'Total_Trades': total_oos_trades,
        'Total_Windows': window_count
    }

if __name__ == "__main__":
    try:
        print("Pre-compiling Numba function for first run optimization...")
        dummy_array = np.array([10.0, 11.0, 10.0])
        # Add dummy_array for volatility argument
        numba_bt(dummy_array, dummy_array, dummy_array, dummy_array, dummy_array, dummy_array,
                 2.0, 0.5, 4.0, 10000.0, 0.0, 0.0, 250, 0.5)
        print("Numba pre-compilation complete.")

        start_time = time.time()
        print(f"fetching data from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}...")
        raw_data = prepare_data(ASSET_A, ASSET_B, START_DATE, END_DATE, SELECTED_TIMEFRAME, SELECTED_TIMEFRAME_VALUE)
        print(f"data acquired successfully. total {SELECTED_TIMEFRAME_VALUE} RTH bars: {len(raw_data)}")

        # 2. **REMOVED** the pre-calculation of Z-Scores/Beta on the full dataset.
        print("Skipping full pre-calculation. Metrics will be calculated inside the WFA loop.")
        processed_full_data = raw_data  # Rename for consistency, but it's the raw data

        # 3. Run Rolling Optimization and Backtest on the pre-processed data
        overall_stats = rolling_bt(
            processed_full_data,
            ROLLING_WINDOW_BARS,
            OPTIMIZATION_WINDOW_BARS,
            ROLLING_SHIFT_DAYS,
            cptl,
            txfee,
            slippage,
            ASSET_A,
            ASSET_B
        )

        end_time = time.time()
        total_time = end_time - start_time

        # 4. Print Higher Order Stats
        if overall_stats:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Timeframe Used: {SELECTED_TIMEFRAME.value}, Value: {SELECTED_TIMEFRAME_VALUE}")
            print(f"ADF Max: {ADF_P_VALUE_THRESHOLD}")
            print(f"Hurst Max: {hurstMax}")
            print(f"Total OOS Windows Tested: {overall_stats['Total_Windows']}")
            print(f"Total Trades Executed: {overall_stats['Total_Trades']}")
            print("!! Final Performance Metrics !!")
            print(f"Total PnL: ${overall_stats['Total_PnL']:,.2f}")
            print(f"annaulized sharpe: {overall_stats['Sharpe_Ratio']:.4f}")
            print(f"annualized calmar: {overall_stats['Calmar_Ratio']:.4f}")
            print(f"MDD: ${overall_stats['Max_Drawdown']:,.2f}")
            print("\n!! Execution Time !!")
            print(f"elapsed time: {total_time:.2f} seconds")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("\nsimulation could not be completed.")

    except Exception as e:
        print(f"\n!!! an error occurred during execution: {e}!!!")
        import traceback


        traceback.print_exc()
