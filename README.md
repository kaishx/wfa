# Algorithmic Pairs Trading: A Dual-Engine Architecture Study

### 1. Introduction

Pairs trading is often presented as the "hello world" of quantitative finance: find two stocks that move together, and when they drift apart, bet on them snapping back. In theory, it's market-neutral and robust. In practice, it's a minefield of regime shifts (correlations breaking) and execution friction (slippage eating profits).

I initiated this project to answer a specific, practical question: **Can a retail algorithm effectively capture alpha on a 15-minute timeframe after accounting for real-world trading costs?**

To answer this, I couldn't just run a simple backtest. I needed a rigorous "stress test" machine—something that could re-optimize itself hundreds of times over years of data without cheating (looking ahead). This is called **Walk-Forward Analysis (WFA)**.

But WFA is computationally expensive. Running thousands of optimization loops takes hours. So, wanting to push my engineering skills, I built a **Dual-Engine Architecture**:

1. **Engine A (The Researcher):** A pure Python version optimized with **Numba**, designed for rapid prototyping and zero-overhead iteration.

2. **Engine B (The Production System):** A high-performance **C++** version linked to Python, designed to simulate a low-latency production environment.

This report documents the journey of building this system, the engineering hurdles of cross-language integration on Windows, and the reality check that market friction imposes on theoretical strategies.

### 2. Methodology: The Strategy Engine

Unlike basic Bollinger Band strategies that rely on fixed averages, I implemented an adaptive system designed to handle changing market conditions.

#### 2.1 The Logic Core

* **Kalman Filter (Dynamic Hedge Ratio):** Markets are not static. A simple moving average lags behind price action. I implemented a Kalman Filter to dynamically calculate the hedge ratio ($\beta$) between two assets, allowing the model to adapt instantly to new price information.

* **Z-Score:** This measures the spread's deviation from its mean, acting as the primary trade signal.

* **Hurst Exponent (The Trend Killer):** This was my "Circuit Breaker." If the Hurst value exceeds 0.6, the spread is trending rather than reverting. The system detects this regime shift and blocks all trades to prevent "catching a falling knife."

* **Dollar-Based Stop Loss:** I calculated stops based on **Gross PnL** (real dollars lost before fees). This introduces path dependency—making the math harder to optimize but far more realistic than simple percentage stops.

#### 2.2 Walk-Forward Analysis (The Stress Test)

To eliminate look-ahead bias, the system uses a rolling window approach:

1. **In-Sample (Train):** The model trains on **120 days** of data to find the optimal Z-Score thresholds ($Z_{entry}, Z_{exit}, Z_{stop}$).

2. **Out-of-Sample (Test):** These parameters are frozen and tested on the next **15 days** of unseen data.

3. **Repeat:** The window slides forward, and the process repeats.

This mimics the real-world constraints of a trader who must make decisions based only on past data.

### 3. Engineering: The Dual-Engine Build

This project was as much about systems engineering as it was about finance. I wanted to compare the two dominant paradigms in quantitative development: Just-In-Time (JIT) compilation vs. Ahead-Of-Time (AOT) compilation.

#### 3.1 Engine A: Python + Numba

This engine uses the `@njit` decorator to compile Python functions into machine code at runtime. It lives entirely inside the Python process, meaning it has zero startup overhead and direct memory access.

#### 3.2 Engine B: C++ + Pybind11

I wrote the core trading logic in standard C++17. I used **Pybind11** to create a bridge, allowing my Python scripts to call the C++ code as if it were a normal library.

* **The Zero-Copy Optimization:** Initially, the C++ engine was slow because it was copying data from Python lists to C++ vectors. I rewrote the interface to use **Zero-Copy Memory Mapping** (`py::array_t`), allowing C++ to read directly from Python's NumPy arrays in RAM without moving a single byte.

#### 3.3 The "DLL Hell" (Windows Integration)

Getting C++ to work seamlessly with Python on Windows was the hardest engineering challenge.

* **The Conflict:** Python on Windows is built with Visual Studio (MSVC), but many tools default to MinGW (GCC). Mixing them caused constant crashes ("DLL load failed").

* **The Solution:** I built a custom `setup.py` build system that automatically detects the correct MSVC compiler and links the necessary runtime libraries (`vcruntime140_1.dll`), creating a portable, pip-installable Python package.

### 4. Discussion: Results & Reality

#### 4.1 Benchmark: JIT vs. AOT

I ran a stress test with **100 Million** data points to compare the two engines.

* **Numba (Python JIT):** ~0.0048 seconds

* **C++ (Compiled):** ~0.0053 seconds

**Analysis:** Numba outperformed the C++ engine by approximately 9%. This counter-intuitive result highlights the cost of the **Foreign Function Interface (FFI)**. For simple, vectorized operations on a single CPU core, the overhead of Python invoking C++ functions exceeds the execution time saved by C++'s raw speed.

* **Implication:** Numba is optimal for research and backtesting due to data locality. However, the C++ architecture remains essential for production environments where logic must execute independently of the Python runtime (e.g., on FPGAs or standalone servers).

#### 4.2 The Friction Barrier

The WFA results underscored the critical impact of trading costs.

* While pairs like `GOOG`/`GOOGL` exhibited high cointegration and stability, the average profit per trade on a 15-minute timeframe hovered around **$2.00**.

* Modeled execution costs (slippage + fees) averaged **~$2.20** per trade.

* **Conclusion:** High-frequency mean reversion on liquid assets is mathematically sound but often unprofitable for retail traders due to transaction friction. Profitability requires sub-penny commission structures or longer holding periods.

#### 4.3 Risk Management Validation

The ADF (Augmented Dickey-Fuller) filter successfully identified regime shifts. For instance, during periods of divergence between `AAPL` and `MSFT`, the system correctly paused trading. This validation confirms the efficacy of the risk management logic, even if net profitability is constrained by costs.

#### 4.4 Visualizing Robustness (Risk vs. Reward)

The visualization generated by the `plotter.py` script (above) plots the **Risk (Max Drawdown)** against the **Reward (Sharpe Ratio)** for all tested pairs across different parameter configurations (ADF/Hurst thresholds).

* **The "Target Zone" (Green Box):** This region represents the ideal profile: high Sharpe (> 1.5) and low Maximum Drawdown (< $500).

* **Observations:**

  * Pairs falling into this zone (e.g., `ALL`/`TRV`) demonstrated consistent mean-reversion behavior capable of overcoming transaction costs.

  * Pairs outside this zone (e.g., `GOOG`/`GOOGL`) were either too volatile (high MDD) or failed to generate enough profit per trade (low Sharpe).

  * The clustering of data points reveals that stricter filtering (e.g., lower ADF p-value thresholds) generally improves the Sharpe Ratio but reduces the total number of trades, highlighting the trade-off between signal quality and opportunity frequency.

### 5. Limitations & Future Work

* **Execution Lag:** The backtest assumes execution at the exact close of the 15-minute candle. In live markets, prices drift in the milliseconds between calculation and execution.

* **Single-Core Speed:** The current system operates on a single CPU thread. Scaling analysis to the full S&P 500 would necessitate a multi-threaded architecture.

* **Platform Dependency:** The build system is currently optimized for Windows. Porting to a Linux-based environment (e.g., Docker) would be required for cloud deployment.

### 6. Conclusion

This project successfully established a professional-grade platform for quantitative research. It demonstrated that while **Walk-Forward Analysis** effectively identifies robust statistical relationships, the **"Cost Barrier"** remains a primary hurdle for retail profitability in high-frequency strategies.

From an engineering standpoint, the project illustrated that **architectural fit** is crucial—the lightweight Numba engine proved superior for research tasks compared to the heavier C++ implementation. However, the development of the C++ infrastructure provides a portable, production-ready foundation capable of bridging the gap between research prototypes and live execution systems.
