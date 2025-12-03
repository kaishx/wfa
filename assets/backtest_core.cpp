#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <cstring>
#include <vector>

namespace py = pybind11;

enum class PositionState : int {
    FLAT = 0,
    LONG = 1,
    SHORT = -1
};

// c++ is essentially black magic to me

double calculate_sharpe_internal(
    const py::detail::unchecked_reference<double, 1>& r_close_a,
    const py::detail::unchecked_reference<double, 1>& r_close_b,
    const py::detail::unchecked_reference<double, 1>& r_z,
    const py::detail::unchecked_reference<double, 1>& r_beta,
    const py::detail::unchecked_reference<double, 1>& r_hurst,
    const py::detail::unchecked_reference<double, 1>& r_vol,

    const py::ssize_t n_bars,

    double z_entry,
    double z_exit,
    double z_stop_loss,
    double capital,
    double txfee_rate,
    double slippage_per_share,
    double abs_dollar_stop,
    double hurst_max,
    double bars_per_year
) {
    PositionState position = PositionState::FLAT;

    double entry_price_a = 0.0;
    double entry_price_b = 0.0;
    double entry_shares_a = 0.0;
    double entry_shares_b = 0.0;

    double w_mean = 0.0;
    double w_m2 = 0.0;
    double w_count = 1.0;

    double total_pnl_check = 0.0;
    const double cost_slippage_factor = slippage_per_share * 2.0;

    for (py::ssize_t i = 0; i < n_bars - 1; ++i) {
        const double current_z = r_z(i);
        const double current_hurst = r_hurst(i);
        double current_step_ret = 0.0;

        if (position != PositionState::FLAT) {
            const double abs_z = std::abs(current_z);
            const bool is_mean_reversion_exit = abs_z <= z_exit;
            const bool is_z_stop_loss_exit = abs_z >= z_stop_loss;
            bool is_dollar_stop_loss_exit = false;

            double next_price_a = r_close_a(i + 1);
            double next_price_b = r_close_b(i + 1);

            if (entry_price_a > 0) {
                 double pnl_a, pnl_b;
                 if (position == PositionState::LONG) {
                    pnl_a = (next_price_a - entry_price_a) * entry_shares_a;
                    pnl_b = (entry_price_b - next_price_b) * entry_shares_b;
                 } else {
                    pnl_a = (entry_price_a - next_price_a) * entry_shares_a;
                    pnl_b = (next_price_b - entry_price_b) * entry_shares_b;
                 }
                 if ((pnl_a + pnl_b) <= -abs_dollar_stop) is_dollar_stop_loss_exit = true;
            }

            const bool is_final_bar = (i == n_bars - 2);

            if (is_mean_reversion_exit || is_z_stop_loss_exit || is_dollar_stop_loss_exit || is_final_bar) {
                 double n_A = entry_shares_a;
                 double n_B = entry_shares_b;

                 double pnl_a, pnl_b;
                 if (position == PositionState::LONG) {
                    pnl_a = (next_price_a - entry_price_a) * n_A;
                    pnl_b = (entry_price_b - next_price_b) * n_B;
                 } else {
                    pnl_a = (entry_price_a - next_price_a) * n_A;
                    pnl_b = (next_price_b - entry_price_b) * n_B;
                 }

                 double gross = pnl_a + pnl_b;
                 double entry_val = entry_price_a * n_A + entry_price_b * n_B;
                 double exit_val = next_price_a * n_A + next_price_b * n_B;
                 double cost = txfee_rate * (entry_val + exit_val) + cost_slippage_factor * (n_A + n_B);

                 double net_pnl = gross - cost;
                 double ret = net_pnl / capital;

                 current_step_ret = ret;
                 total_pnl_check += std::abs(net_pnl);

                 position = PositionState::FLAT;
                 entry_price_a = 0.0; entry_price_b = 0.0;
                 entry_shares_a = 0.0; entry_shares_b = 0.0;
            }
        }
        else if (position == PositionState::FLAT && i != n_bars - 2) {
            if (std::abs(current_z) >= z_entry && current_hurst < hurst_max) {
                 double beta_val = r_beta(i);
                 if (std::isnan(beta_val)) {
                 } else {
                     beta_val = std::abs(beta_val);
                     double current_vol = r_vol(i);
                     double vol_scale = 1.0;
                     if (current_vol > 0) {
                         vol_scale = 0.15 / current_vol;
                         if (vol_scale > 1.0) vol_scale = 1.0;
                     }
                     double adj_capital = capital * vol_scale;
                     double next_price_a = r_close_a(i + 1);
                     double next_price_b = r_close_b(i + 1);

                     double V_A = adj_capital / (1.0 + beta_val);
                     double V_B = adj_capital - V_A;

                     entry_shares_a = std::floor(V_A / next_price_a);
                     entry_shares_b = std::floor(V_B / next_price_b);
                     entry_price_a = next_price_a;
                     entry_price_b = next_price_b;

                     if (current_z < -z_entry) position = PositionState::LONG;
                     else position = PositionState::SHORT;
                 }
            }
        }


        w_count += 1.0;
        double delta = current_step_ret - w_mean;
        w_mean += delta / w_count;
        double delta2 = current_step_ret - w_mean;
        w_m2 += delta * delta2;
    }

    if (total_pnl_check == 0.0) return 0.0;
    if (w_m2 <= 1e-16) return 0.0;

    double std_dev = std::sqrt(w_m2 / w_count);

    if (std_dev == 0.0) return 0.0;
    return (w_mean * bars_per_year) / (std_dev * std::sqrt(bars_per_year));
}

py::array_t<double> run_pairs_backtest(
    py::array_t<double> const& close_a,
    py::array_t<double> const& close_b,
    py::array_t<double> const& z_score,
    py::array_t<double> const& beta,
    py::array_t<double> const& hurst,
    py::array_t<double> const& volatility,
    double z_entry, double z_exit, double z_stop_loss, double capital,
    double txfee_rate, double slippage_per_share, double abs_dollar_stop, double hurst_max
) {
    auto r_close_a = close_a.unchecked<1>();
    auto r_close_b = close_b.unchecked<1>();
    auto r_z = z_score.unchecked<1>();
    auto r_beta = beta.unchecked<1>();
    auto r_hurst = hurst.unchecked<1>();
    auto r_vol = volatility.unchecked<1>();

    if (close_a.size() != close_b.size() ||
        close_a.size() != z_score.size() ||
        close_a.size() != beta.size() ||
        close_a.size() != hurst.size() ||
        close_a.size() != volatility.size())
    {
        throw std::runtime_error("Input array size mismatch in run_pairs_backtest. All data arrays must be the same length.");
    }

    const py::ssize_t n_bars = r_close_a.shape(0);
    auto pnl_array = py::array_t<double>(n_bars);
    if (n_bars < 2) return pnl_array;

    std::memset(pnl_array.mutable_data(), 0, n_bars * sizeof(double));
    auto w_pnl = pnl_array.mutable_unchecked<1>();

    PositionState position = PositionState::FLAT;
    double entry_price_a = 0.0, entry_price_b = 0.0, entry_shares_a = 0.0, entry_shares_b = 0.0;
    const double cost_slippage_factor = slippage_per_share * 2.0;

    for (py::ssize_t i = 0; i < n_bars - 1; ++i) {
        double current_z = r_z(i);
        double current_hurst = r_hurst(i);
        double next_price_a = r_close_a(i+1);
        double next_price_b = r_close_b(i+1);

        if (position != PositionState::FLAT) {
             const double abs_z = std::abs(current_z);
             bool exit = (abs_z <= z_exit) || (abs_z >= z_stop_loss) || (i == n_bars - 2);

             if (!exit && entry_price_a > 0) {
                 double pa, pb;
                 if (position == PositionState::LONG) {
                    pa = (next_price_a - entry_price_a) * entry_shares_a;
                    pb = (entry_price_b - next_price_b) * entry_shares_b;
                 } else {
                    pa = (entry_price_a - next_price_a) * entry_shares_a;
                    pb = (next_price_b - entry_price_b) * entry_shares_b;
                 }
                 if ((pa + pb) <= -abs_dollar_stop) exit = true;
             }

             if (exit && entry_price_a > 0) {
                 double nA = entry_shares_a, nB = entry_shares_b;
                 double pnl_a, pnl_b;

                 if (position == PositionState::LONG) {
                    pnl_a = (next_price_a - entry_price_a) * nA;
                    pnl_b = (entry_price_b - next_price_b) * nB;
                 } else {
                    pnl_a = (entry_price_a - next_price_a) * nA;
                    pnl_b = (next_price_b - entry_price_b) * nB;
                 }

                 double gross = pnl_a + pnl_b;
                 double entry_val = entry_price_a * nA + entry_price_b * nB;
                 double exit_val = next_price_a * nA + next_price_b * nB;
                 double cost = txfee_rate * (entry_val + exit_val) + cost_slippage_factor * (nA + nB);

                 w_pnl(i+1) = gross - cost;
                 position = PositionState::FLAT;
                 entry_price_a = 0.0; entry_price_b = 0.0;
                 entry_shares_a = 0.0; entry_shares_b = 0.0;
             }
        } else if (i != n_bars - 2) {
            if (std::abs(current_z) >= z_entry && current_hurst < hurst_max) {
                 double b = r_beta(i);
                 if (!std::isnan(b)) {
                     b = std::abs(b);

                     double current_vol = r_vol(i);
                     double vol_scale = 1.0;
                     if (current_vol > 0) {
                         vol_scale = 0.15 / current_vol;
                         if (vol_scale > 1.0) vol_scale = 1.0;
                     }
                     double adj_capital = capital * vol_scale;

                     double Va = adj_capital / (1.0 + b);
                     entry_shares_a = std::floor(Va / next_price_a);
                     entry_shares_b = std::floor((adj_capital - Va) / next_price_b);
                     entry_price_a = next_price_a;
                     entry_price_b = next_price_b;

                     if (current_z < -z_entry) position = PositionState::LONG;
                     else position = PositionState::SHORT;
                 }
            }
        }
    }
    return pnl_array;
}

py::array_t<double> run_optimization_core(
    py::array_t<double> const& close_a,
    py::array_t<double> const& close_b,
    py::array_t<double> const& z_score,
    py::array_t<double> const& beta,
    py::array_t<double> const& hurst,
    py::array_t<double> const& volatility,
    py::array_t<double> const& z_entry_grid,
    py::array_t<double> const& z_exit_grid,
    py::array_t<double> const& z_sl_grid,
    double capital, double txfee, double slippage, double abs_stop, double hurst_max, double bars_per_year
) {
    auto r_close_a = close_a.unchecked<1>();
    auto r_close_b = close_b.unchecked<1>();
    auto r_z = z_score.unchecked<1>();
    auto r_beta = beta.unchecked<1>();
    auto r_hurst = hurst.unchecked<1>();
    auto r_vol = volatility.unchecked<1>();

    if (close_a.size() != close_b.size() ||
        close_a.size() != z_score.size() ||
        close_a.size() != beta.size() ||
        close_a.size() != hurst.size() ||
        close_a.size() != volatility.size())
    {
        throw std::runtime_error("Input array size mismatch in run_pairs_backtest. All data arrays must be the same length.");
    }

    auto r_g_entry = z_entry_grid.unchecked<1>();
    auto r_g_exit = z_exit_grid.unchecked<1>();
    auto r_g_sl = z_sl_grid.unchecked<1>();

    py::ssize_t n_bars = r_close_a.shape(0);

    double best_sharpe = -std::numeric_limits<double>::infinity();
    double best_e = 0.0, best_x = 0.0, best_sl = 0.0;
    bool found = false;

    for (py::ssize_t i = 0; i < r_g_entry.shape(0); ++i) {
        double e = r_g_entry(i);
        for (py::ssize_t j = 0; j < r_g_exit.shape(0); ++j) {
            double x = r_g_exit(j);
            for (py::ssize_t k = 0; k < r_g_sl.shape(0); ++k) {
                double sl = r_g_sl(k);

                if (!(x < e && e < sl)) continue;

                double sharpe = calculate_sharpe_internal(
                    r_close_a, r_close_b, r_z, r_beta, r_hurst, r_vol,
                    n_bars,
                    e, x, sl, capital, txfee, slippage, abs_stop, hurst_max, bars_per_year
                );

                if (sharpe > best_sharpe) {
                    best_sharpe = sharpe;
                    best_e = e; best_x = x; best_sl = sl;
                    found = true;
                }
            }
        }
    }

    py::array_t<double> result(4);
    auto w_res = result.mutable_unchecked<1>();
    if (found) {
        w_res(0) = best_e; w_res(1) = best_x; w_res(2) = best_sl; w_res(3) = best_sharpe;
    } else {
        w_res(0) = 0.0; w_res(1) = 0.0; w_res(2) = 0.0; w_res(3) = -100.0;
    }
    return result;
}

PYBIND11_MODULE(cpp_accelerator, m) {
    m.doc() = "C++ Optimization Core";
    m.def("run_backtest", &run_pairs_backtest, "Standard backtest returning PnL array");
    m.def("run_optimization_core", &run_optimization_core, "Full grid search internal to C++");
}