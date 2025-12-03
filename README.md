# Algorithmic Pairs Trading: A Dual-Engine Architecture Study

*This is written informally and is intended to be more blog-like so I never make formal references.*

*Link back to my [Portfolio (github)](https://kaishx.github.io/#projects)*

## 1. Introduction & Thesis

In 2024, I was introduced to a form of pseudo-gambling on the stock market by a friend. It was the two 3x leveraged semiconductor ETFs, SOXL and SOXS, which were incredibly volatile in that period, and could see massive downs and ups then. While watching them on my stock app, I noticed that the tracking within them was not perfect - in essence, one could have moved by 3% while the other, 2.95%. 

I thought I found a hack which would exploit this difference, and I came up with my own trading strategy to arbitrage my market position (make it delta neutral) while betting on the fact the absolute movements would meet each other again. Alas, 1 week in, I did my first good Google search, and that was when I found out my "exploitation" had already been done for many decades - Pairs Trading.

Pairs trading is often presented as the "hello world" of quantitative finance: find two stocks that move together, and when they drift apart, bet on them snapping back. In theory, it's **market-neutral** and robust. But realistically, there are slippage, transactions costs, and correlation breakdowns.

Hence, I started this project to get my answers to a specific, practical question: **Can a retail algorithm effectively capture alpha on a 15-minute timeframe after accounting for real-world trading costs?**

To answer this, I couldn't just run a simple backtest which would just be overfit and give me unrealistic results. I needed a rigorous "stress test" machine—something that could re-optimize itself hundreds of times over years of data without cheating (looking ahead). This is called **Walk-Forward Analysis (WFA)**.

WFA is computationally expensive. Running thousands of optimization loops takes hours. So, wanting to push my engineering skills, I built a WFA Engine with two different styles: 

1. A Python-only, **Numba**-optimized Engine to reduce overhead and without the complexities of setting up a **C++** Module
2. A **C++** Module which is integrated into the Python Engine for maximum computational speed and high performance.

---

## 2. Strategy & Methodology (WFA)

Unlike basic strategies that rely on fixed averages, I utilized many statistical tests to adapt to volatile market conditions

### The Logic Core

* **Kalman Filter (Dynamic Hedge Ratio):** Implemented a Kalman Filter to dynamically calculate the hedge ratio ($\beta$) between two assets, allowing the model to adapt instantly to new price information, thus avoiding the lag inherent in simple moving averages.
* **Z-Score:** Measures the spread's deviation from its mean, which acts as the primary trade signal. The system enters a trade if the magnitude of the current Z-score, $|\mathbf{Z_{current}}|$, exceeds the entry threshold, $\mathbf{Z_{entry}}$.
* **Augmented Dickey-Fuller Test (ADF):** **1st Guard** The test is calculated on the preceding In-Sample (IS) window to confirm **stationarity** (cointegration). If the **P-value** of the test exceeds a predefined threshold (e.g., $P$-value $> 0.20$), the IS window is deemed non-stationary, and the subsequent Out-of-Sample (OOS) window will not be traded.
* **Hurst Exponent:** **2nd Guard** Measures the degree of **mean reversion** versus **trending behavior** behaviour in the spread. If the Hurst Value exceeds a threshold (e.g., $H > 0.75$), the system detects persistent, trending behavior and blocks the trade at that specific moment.
* **Dollar-Based Stop Loss:** Calculated stops based on **Gross PnL** (real dollars lost before fees), making the optimization more path-dependent and realistic than simple percentage stops.

**Note:** With the last three components, we heavily reduce trading risk by ensuring that market positions are only initiated when both **statistical confidence (ADF)** and **current behavior (Hurst)** strongly favor mean reversion.


### Walk-Forward Analysis (The Stress Test)

To eliminate look-ahead bias, the system uses a rolling window approach:

1.  **In-Sample (Train):** The model trains on **120 days** of data to find the optimal Z-Score thresholds ($Z_{entry}, Z_{exit}, Z_{stop}$).
2.  **Out-of-Sample (Test):** These parameters are frozen and tested on the next **15 days** of unseen data.
3.  **Repeat:** The window slides forward, and the process repeats, mimicking real-world constraints.

---

## 3. Engineering: Dual-Engine Build

This project focused on comparing the two dominant paradigms in quantitative development: **Just-In-Time (JIT) compilation** (Numba) vs. **Ahead-Of-Time (AOT) compilation** (C++).

### Engine B: C++ + Pybind11 (The Core Accelerator)

* **The Zero-Copy Optimization:** I rewrote the data interface using **Zero-Copy Memory Mapping** (`py::array_t`). This allows the C++ core to read directly from Python's NumPy arrays in RAM **without copying data**, which eliminates a major source of latency.
* **The "DLL Hell" Solution (Windows Integration):** Solved complex Windows dependency issues (MSVC vs. GCC conflict) by building a custom `setup.py` that automatically detects the correct MSVC compiler and links the necessary runtime libraries (`vcruntime140_1.dll`), resulting in a **portable, pip-installable Python package**.

---

## 4. Installation & Build Guide

### Prerequisites

* **Python 3.10+**
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
    pip install . --force-reinstall
    ```

### Configuration and Security

* Create a file named **`.env`** in the root directory to securely store your Alpaca API credentials.
    ```env
    # .env file content (Add this file to .gitignore)
    ALPACA_KEY_ID="YOUR_API_KEY_HERE"
    ALPACA_SECRET_KEY="YOUR_SECRET_KEY_HERE"
    ```

---

## 5. Usage & Workflow

The analysis runs in three simple steps:

1.  **Define Parameters:** Customize the asset pairs and parameters in **`pairs_configs.json`**.
2.  **Run WFA:** Execute the batch runner to initiate all backtests (uses the C++ core).
    ```bash
    python batch_runner.py
    ```
3.  **Analyze & Visualize:** Process the log files (P.S: remember to point to the right directories!) to generate a master report and plots.
    ```bash
    python all_in_one.py
    python plotter.py
    ```

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
