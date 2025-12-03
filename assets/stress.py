import os
import sys
import time
import numpy as np
import pandas as pd
from numba import njit
import plotly.graph_objects as go
from scipy.stats import norm

if os.name == 'nt':
    try:
        os.add_dll_directory(os.getcwd())
    except Exception:
        pass

try:
    import cpp_accelerator

    print("C++ module loaded")
except ImportError as e:
    print(f"failed to load C++ Module: {e}")
    sys.exit(1)

tosimulate = 10_000  # yeh not sure why the number is in underscores. 10k is similar to the wfa data size so leave it like this
# found that theoretically, lower bars is better for C++
num_runs = 1000
cptl = 10000.0
txfee = 0.0001
slippage = 0.01
ABS_STOP = 250.0
hustMax = 0.75
BARS_PER_YEAR = 252 * 26

Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 3.0

# mock grid for stress test. more grid = C++ gets more advantage on numba.
Z_ENTRY_GRID = np.linspace(1.5, 3.0, 3)
Z_EXIT_GRID = np.linspace(0.0, 1.0, 3)
Z_STOP_GRID = np.linspace(3.5, 5.0, 3)


# numba part, kindaaa copied straight over from numba_wfa and changed sum stuff
@njit
def numba_bt(close_a, close_b, z_score, beta, hurst, z_entry, z_exit, z_stop_loss, cptl, txfee, slippage,
                       abs_stop, hurstMax):
    n = len(close_a)
    pnl = np.zeros(n)
    pos = 0.0
    ep_a = 0.0;
    ep_b = 0.0;
    es_a = 0.0;
    es_b = 0.0

    for i in range(n - 1):
        z = z_score[i]
        h = hurst[i]
        prev_pos = pos

        np_a = close_a[i + 1]
        np_b = close_b[i + 1]
        b_val = beta[i]

        # Exit
        if prev_pos != 0.0:
            is_mr = abs(z) <= z_exit
            is_sl = abs(z) >= z_stop_loss
            is_dsl = False

            if ep_a > 0:
                if prev_pos == 1.0:
                    pa = (np_a - ep_a) * es_a
                    pb = (ep_b - np_b) * es_b
                else:
                    pa = (ep_a - np_a) * es_a
                    pb = (np_b - ep_b) * es_b
                if (pa + pb) <= -abs_stop: is_dsl = True

            if is_mr or is_sl or is_dsl or (i == n - 2):
                if ep_a > 0:
                    if prev_pos == 1.0:
                        pa = (np_a - ep_a) * es_a
                        pb = (ep_b - np_b) * es_b
                    else:
                        pa = (ep_a - np_a) * es_a
                        pb = (np_b - ep_b) * es_b

                    gross = pa + pb
                    entry_v = ep_a * es_a + ep_b * es_b
                    exit_v = np_a * es_a + np_b * es_b
                    cost = (txfee * (entry_v + exit_v)) + (slippage * (es_a + es_b) * 2.0)
                    pnl[i + 1] = gross - cost
                    pos = 0.0
                    ep_a = 0.0;
                    ep_b = 0.0;
                    es_a = 0.0;
                    es_b = 0.0

        # Entry
        elif prev_pos == 0.0 and (i != n - 2):
            if abs(z) >= z_entry and h < hurstMax:
                if np.isnan(b_val): continue
                bv = abs(b_val)
                va = cptl / (1.0 + bv)
                vb = cptl - va
                na = va / np_a
                nb = vb / np_b

                ep_a = np_a;
                ep_b = np_b
                es_a = na;
                es_b = nb
                if z < -z_entry:
                    pos = 1.0
                else:
                    pos = -1.0

    return pnl


def run_opt(close_a, close_b, z_score, beta, hurst, volatility, grids):
    best_sharpe = -np.inf

    for z_entry in grids[0]:
        for z_exit in grids[1]:
            for z_stop in grids[2]:
                if not (z_exit < z_entry < z_stop): continue

                pnl = numba_bt(
                    close_a, close_b, z_score, beta, hurst,
                    z_entry, z_exit, z_stop, cptl, txfee, slippage, ABS_STOP, hustMax
                )

                if np.sum(np.abs(pnl)) != 0:
                    ret = pnl / cptl
                    m = np.mean(ret)
                    s = np.std(ret)
                    if s > 0:
                        sharpe = (m / s) * np.sqrt(BARS_PER_YEAR)
                        if sharpe > best_sharpe: best_sharpe = sharpe


def makeData(n):
    print(f"\ngenerating {n:,} bars of mock data...")
    np.random.seed(42)
    r1 = np.random.normal(0, 0.01, n)
    r2 = np.random.normal(0, 0.01, n)
    close_a = 100 * np.exp(np.cumsum(r1))
    close_b = 200 * np.exp(np.cumsum(r2))

    z_score = np.random.normal(0, 1.5, n)
    beta = np.random.normal(1.0, 0.1, n)
    hurst = np.random.uniform(0.3, 0.9, n)
    volatility = np.random.uniform(0.01, 0.2, n)

    return (
        close_a.astype(np.float64),
        close_b.astype(np.float64),
        z_score.astype(np.float64),
        beta.astype(np.float64),
        hurst.astype(np.float64),
        volatility.astype(np.float64)
    )


# love that 3b1b look
def plot(numba_times, cpp_times, title):
    fig = go.Figure()

    mu_numba, std_numba = norm.fit(numba_times)
    mu_cpp, std_cpp = norm.fit(cpp_times)

    min_x = min(min(numba_times), min(cpp_times))
    max_x = max(max(numba_times), max(cpp_times))
    x_axis = np.linspace(min_x, max_x, 1000)

    pdf_numba = norm.pdf(x_axis, mu_numba, std_numba)
    pdf_cpp = norm.pdf(x_axis, mu_cpp, std_cpp)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=pdf_numba,
        mode='lines',
        name=f'Numba (μ={mu_numba:.5f}s)',
        line=dict(color='cyan', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))

    # C++ Curve
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=pdf_cpp,
        mode='lines',
        name=f'C++ (μ={mu_cpp:.5f}s)',
        line=dict(color='orange', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.1)'
    ))

    fig.add_vline(x=mu_numba, line_dash="dash", line_color="cyan", opacity=0.5)
    fig.add_vline(x=mu_cpp, line_dash="dash", line_color="orange", opacity=0.5)

    fig.update_layout(
        title=f"{title}: numba vs C++ performance dist: ({num_runs} runs @ {tosimulate:,} bars)",
        xaxis_title="execution time in (s)",
        yaxis_title="probability density",
        template="plotly_dark",
        height=700,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    output_path = f"results.html"
    fig.write_html(output_path, auto_open=False)
    print(f"\n saved graph to: {output_path}")


if __name__ == "__main__":
    ca, cb, zs, be, hu, vol = makeData(tosimulate)

    # 1. OPTIMIZATION LOOP STRESS TEST
    print("\n--- STRESS TEST 1: OPTIMIZATION LOOP ---")
    g_e_arr = Z_ENTRY_GRID.astype(np.float64)
    g_x_arr = Z_EXIT_GRID.astype(np.float64)
    g_s_arr = Z_STOP_GRID.astype(np.float64)

    total_combinations = len(Z_ENTRY_GRID) * len(Z_EXIT_GRID) * len(Z_STOP_GRID)
    print(f"Combinations: {total_combinations}")

    # Warmup
    numba_bt(ca[:100], cb[:100], zs[:100], be[:100], hu[:100], 2.0, 0.5, 3.0, cptl, txfee, slippage, ABS_STOP,
                       hustMax)

    numba_opt_times = []
    cpp_opt_times = []

    print(f"testing {tosimulate:,} bars, repeated {num_runs} times.")

    for run in range(num_runs):
        if run % 5 == 0: print(f"Run {run}/{num_runs}...", end='\r')

        t0 = time.perf_counter()
        run_opt(ca, cb, zs, be, hu, vol,[Z_ENTRY_GRID, Z_EXIT_GRID, Z_STOP_GRID])
        t1 = time.perf_counter()
        numba_opt_times.append(t1 - t0)

        t0 = time.perf_counter()
        cpp_accelerator.run_optimization_core(
            ca, cb, zs, be, hu, vol,
            g_e_arr, g_x_arr, g_s_arr,
            cptl, txfee, slippage, ABS_STOP, hustMax, float(BARS_PER_YEAR)
        )
        t1 = time.perf_counter()
        cpp_opt_times.append(t1 - t0)

    print(f"Run {num_runs}/{num_runs} complete.")

    numba_avg = np.mean(numba_opt_times)
    cpp_avg = np.mean(cpp_opt_times)
    numba_std = np.std(numba_opt_times)
    cpp_std = np.std(cpp_opt_times)

    print(f"numba Avg={numba_avg:.6f}s | std={numba_std:.6f}")
    print(f"c++   Avg={cpp_avg:.6f}s | std={cpp_std:.6f}")

    if cpp_avg < numba_avg:
        diff = (numba_avg / cpp_avg)
        print(f"c++ is {diff:.2f}x faster than numba")
    else:
        diff = (cpp_avg / numba_avg)
        print(f"numba is {diff:.2f}x faster than c++")

    plot(numba_opt_times, cpp_opt_times, "Optimization Loop")