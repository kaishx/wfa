# Algorithmic Pairs Trading: A Dual-Engine Architecture Study

*This is written informally and is intended to be more blog-like so I never make formal references.*

*Link back to my [Portfolio (github)](https://kaishx.github.io/#projects)*

## 1. Introduction & Thesis

In 2024, I was introduced to a form of pseudo-gambling on the direction of the stock market by a friend. It was the two 3x leveraged semiconductor ETFs, SOXL & SOXS, which were extremely volatile products that could swing aggressively in either direction. While watching them on my stock app, I noticed something interesting: the tracking between the bullish and bearish versions wasn't perfectly aligned. One might move 3% while the other moved 2.95%.

I thought I found a hack that would exploit this difference, and I came up with my own trading strategy to hedge my market position while betting on the fact the absolute movements would meet each other again. Alas, 1 week in, I did my first *good* Google search, and that was when I found out my "hack" already had a name - Pairs Trading.

Pairs trading is often presented as the "hello world" of quantitative finance: find two stocks that move together, and when they drift apart, bet on them snapping back. In theory, it's **market-neutral** and robust. But realistically, there are slippage, transactions costs, and correlation breakdowns.

Hence, I started this project to get my answers to a specific, practical question: **Can a retail algorithm effectively capture alpha on a 15-minute timeframe after accounting for real-world trading costs in 2025?**

To answer this, I couldn't just run a simple backtest which would just be overfit and give me unrealistic results. I needed a rigorous "stress test" machine—something that could re-optimize itself hundreds of times over years of data without cheating (looking ahead). This is called **Walk-Forward Analysis (WFA)**.

WFA is computationally expensive. Running thousands of optimization loops takes hours. So, wanting to push my engineering skills, I built a WFA Engine with two different optimization methods: 

1. A Python-only, **Numba**-optimized Engine to reduce overhead and without the complexities of setting up a **C++** Module
2. A **C++** Module which is integrated into the Python Engine for maximum computational speed and high performance.

---

## 2. Methodology (WFA) 
Related Code: (`cpp_wfa.py` and `numba_wfa.py`)

Unlike basic strategies that rely on fixed averages, I utilized many statistical tests to adapt to volatile market conditions.

### Logic

* **Kalman Filter (Dynamic Hedge Ratio):** Implemented a Kalman Filter to dynamically calculate the hedge ratio ($\beta$) between two assets, allowing the model to adapt instantly to new price information, thus avoiding the lag inherent in simple moving averages.
* **Z-Score:** Measures the spread's deviation from its mean, which acts as the primary trade signal. The system enters a trade if the magnitude of the current Z-score, $|\mathbf{Z_{current}}|$, exceeds the entry threshold, $\mathbf{Z_{entry}}$.
* **Augmented Dickey-Fuller Test (ADF):** **1st Guard** The test is calculated on the preceding In-Sample (IS) window to confirm **stationarity** (cointegration). If the **P-value** of the test exceeds a predefined threshold (e.g., $P$-value $> 0.20$), the IS window is deemed non-stationary and the OOS window is skipped entirely.
* **Hurst Exponent:** **2nd Guard** Measures the degree of **mean reversion behavior** versus **trending behavior** in the spread. If the Hurst Value exceeds a threshold (e.g., $H > 0.75$), the system detects persistent, trending behavior and blocks the trade at that specific moment.
* **Dollar-Based Stop Loss:** Calculated stops based on **Gross PnL** (real dollars lost before fees), making the optimization more path-dependent and realistic than simple percentage stops.

**Note:** With these components, we heavily reduce trading risk by ensuring that market positions are only initiated when both **statistical confidence (ADF)** and **current behavior (Hurst)** strongly favor mean reversion. The **Dollar-Based Stop Loss** acts as an absolute risk ceiling, protecting capital during black-swan events or rapid mean reversion failures.


### WFA Process 

I implemented a rolling-window approach to eliminate lookahead bias, and the WFA process follows as such:

1.  **In-Sample (Train):** The model trains on **60 days** of data to find the optimal Z-Score thresholds ($Z_{entry}, Z_{exit}, Z_{stop}$). The window is blocked if it exceeds the above mentioned ADF threshold.
2.  **Out-of-Sample (Test):** These parameters are frozen and tested on the next **15 days** of unseen data. Any trades during the OOS are blocked if its Hurst Exponent exceeds the above mentioned Hurst threshold.
3.  **Repeat:** The window slides forward, and the process repeats, mimicking real-world constraints.

---

## 3. Methodology (C++/Numba Comparison) 
Related Code: (`stress.py`)

Initially, my WFA engine was extremely slow—Python for loops were simply not capable of handling the thousands of optimization cycles required for each rolling window. To address this, I first migrated the bottleneck logic into a Numba JIT-compiled function, which significantly improved performance by removing much of Python’s overhead.

However, after seeing the complexity and scale of the project grow, I wanted even more speed and consistency. This led me to develop a separate C++ module, integrated back into Python, to push the optimization performance further.

Hence, the latter part of this project therefore focuses on comparing these two acceleration strategies—Numba JIT vs. the C++ module—for the core optimization loop inside the Walk-Forward Analysis (WFA) engine.

The main computational bottleneck of WFA occurs in the optimization and backtest loops. For each in-sample window, the engine iterates through hundreds of parameter combinations and executes the backtest logic thousands of times. To evaluate the performance benefit of the C++ module, I designed a controlled stress test to compare both implementations.

### Stress Test Process

* **Data Generation**: Generated ~10,000 bars of mock time-series data (e.g prices, Z-scores, Hurst exponents) to mimic data that the WFA code would have worked on.
* **Test Scope**: The test focused exclusively on the slowest segment of the entire WFA workflow: the Optimization and Backtest Loops. The loop involves iterating through hundreds of parameter combinations (the Z-score grids) and running the backtest logic for each one.
* **Controlled Execution**: The optimization run was performed 1,000 times against the same mock data for both the Numba engine and the C++ accelerator to build a statistically valid sample of execution times.
* **C++ Test**: The C++ side utilizes the exposed run_optimization_core function, which handles iterating through the Z-score grids and calculating the Sharpe Ratio within its native C++ environment for maximum speed.
* **Numba Test**: The Numba side runs the iteration process directly in Python using itertools.product and calls the @njit decorated numba_bt function for the backtesting logic.

---

## 4. Installation & Build Guide

### Prerequisites

* **Python 3.11**
* **C++ Compiler:** MSVC (Windows), or GCC/Clang (Linux/macOS).
* **Dependencies:** Install all Python packages using the provided file:

    ```bash
    pip install -r requirements.txt
    ```

### Building the C++ Accelerator

The high-performance core must be compiled for your system:

1.  Navigate to the root directory containing `setup.py`.
2.  Run the build command:

    ```bash
    pip install . --no-build-isolation --force-reinstall
    ```

This will compile and install the C++ extension module on your system.

### Configuration and Security

* Create a file named **`.env`** in the root directory to securely store your Alpaca API credentials.

    ```env
    ALPACA_KEY_ID="YOUR_API_KEY_HERE"
    ALPACA_SECRET_KEY="YOUR_SECRET_KEY_HERE"
    ```
---

## 5. Usage & Workflow

To run either the C++-accelerated WFA or the Python/Numba version, follow three steps:

1. Configure Your Run

* Set parameters such as hurstMax inside the WFA script.

* Open batch_runner.py and update the wfa_script variable to point to either:

    * `"cpp_wfa"` for the C++ engine

    * `"numba_wfa"` for the Numba engine

* Add your desired stock pairs and configurations using the template provided inside batch_runner.py.

2.  **Run WFA:** Execute the batch runner to initiate all backtests (uses the C++ core).
    ```bash
    python batch_runner.py
    ```
3.  **Analyze & Visualize:** Process the log files (P.S: remember to point to the right directories!) to generate a master report and plots.
    ```bash
    python all_in_one.py
    python plotter.py
    ```
    
If you have trouble with the C++ for whatever reason, you may opt to use the python-only Numba engine which is only very slightly slower. Simply ensure the `wfa_script` in `batch_runner.py` points to `numba_wfa` instead.

---

## 6. Benchmark: C++ vs. Numba Performance

### Benchmark Results (1,000 runs @ 10,000 bars)

* **C++ (Engine B):** $\mu \approx 0.0071s$
* **Numba (Engine A):** $\mu \approx 0.0167s$



[Image of Benchmark Distribution Graph]


**Analysis:** The C++ engine was **~2.35x faster** than the optimized Numba version.

* **Raw Speed:** C++ eliminated Python overhead during the heavy grid-search loops, leveraging the Zero-Copy technique for superior execution speed.
* **Consistency (The Hidden Winner):** The C++ performance distribution is much tighter than the Numba curve. In production, **predictable latency** is crucial, which the C++ engine delivers by being immune to Python's Garbage Collection overhead.

---

## 7. Discussion & Financial Reality

### The Friction Barrier

WFA results showed that while pairs like `GOOG`/`GOOGL` are highly stable, the average profit per trade was **~$2.00**, while modeled execution costs (slippage + fees) averaged **~$2.20**.

* **Conclusion:** High-frequency mean reversion is mathematically sound but often **unprofitable for retail traders** due to high transaction friction.

### Risk Management Validation

* The Hurst Exponent successfully identified and filtered out trending market regimes.
* The ADF filter confirmed cointegration validity before optimization, ensuring the system traded only on statistically robust relationships.

### Visualizing Robustness (Risk vs. Reward)

The Plotly visualization charts **Risk (Max Drawdown)** against the **Reward (Sharpe Ratio)** for all tested pairs.

* **Observations:** The clustering of data points revealed that stricter filtering (e.g., lower ADF p-value thresholds) generally improves the Sharpe Ratio but reduces the total number of trades, highlighting the trade-off between signal quality and opportunity frequency.


* **The Zero-Copy Optimization:** I rewrote the data interface using **Zero-Copy Memory Mapping** (`py::array_t`). This allows the C++ core to read directly from Python's NumPy arrays in RAM **without copying data**, which eliminates a major source of latency.
* **The "DLL Hell" Solution (Windows Integration):** Solved complex Windows dependency issues (MSVC vs. GCC conflict) by building a custom `setup.py` that automatically detects the correct MSVC compiler and links the necessary runtime libraries (`vcruntime140_1.dll`), resulting in a **portable, pip-installable Python package**.

---

## 8. Limitations & Future Work

* **Execution Lag:** The backtest assumes execution at the exact candle close, which neglects live market execution delays.
* **Single-Core Speed:** The system currently runs on a single thread. Scaling analysis to larger universes requires a multi-threaded C++ implementation.
* **Platform Dependency:** Future work requires porting the custom Windows build system to a Linux/Docker environment for cloud deployment readiness.

---

## 9. Conclusion

This project successfully established a professional-grade quantitative research platform. It demonstrated that while **Walk-Forward Analysis** confirms robust trading relationships, the **"Cost Barrier"** remains a primary hurdle for retail profitability.

The benchmark proved that while Numba is excellent for research, **C++ remains the king of predictable production performance**. The development of this hybrid architecture provides a portable, high-performance foundation capable of bridging the gap between research prototypes and live execution systems.

## Timeline of my project

- Started running my first backtests on Google Colab, which could run 1 backtest per night with its timeouts
- Moved on to my computer to utilize my computer's strong hardware, bumping up my backtests to 10 per night
- Realized my backtests were so overfitted and realized that I had unrealistic results, so I made my first WFA
- I could only expect to see the WFA for one pair done every 10 minutes
- Improved the WFA with Numba, seeing it drop to over 60 runs per hour
- Realized I was downloading the same data everytime, and implemented caching of data, allowing me to make one WFA run in under 30 seconds by avoiding downloading.
- Wanted to optimize improvements, and hence added C++ modules to expensive calculation functions, allowing me to run one WFA in under 10 seconds.
