# An Exploration into Pairs Trading, 40 Years Later...

*This is written informally and is intended to be more blog-like so I never make formal references.*

*Link back to my [Portfolio (github)](https://kaishx.github.io/#projects)*

## Overview

The project implements a Walk-Forward Analysis (WFA) for pairs trading with two acceleration paths: a Numba-optimized Python engine and a native C++ module. It performs rolling-window optimization, runs stationarity and mean-reversion checks, and executes millions of backtest iterations. I also built stress tests to compare Numba vs. native C++ performance under realistic loads.

## 1. Introduction & Thesis

In 2024, I was introduced to a form of pseudo-gambling on the direction of the stock market by a friend. It was the two 3x leveraged semiconductor ETFs, SOXL & SOXS, which were extremely volatile products that could swing aggressively in either direction. 

While watching them on my stock app, I noticed something interesting: the tracking between the bullish and bearish versions wasn't perfectly aligned. One might move 3% while the other moved 2.95%.

I thought I found a hack that would exploit this difference, and I came up with my own trading strategy to hedge my market position while betting on the fact the absolute movements would meet each other again. Alas, 1 week in, I did my first *good* Google search, and that was when I found out my "hack" already had a name - Pairs Trading.

Pairs trading is one of the most well-known and widely researched topics of quantitative finance: find two stocks that move together, and when they drift apart, bet on them snapping back. Theoretically, it's market-neutral and robust. But realistically, there are slippage, transactions costs, and correlation breakdowns.

Hence, I started this project to get my answers to a specific, practical question: **Can a retail algorithm effectively capture alpha on a 15-minute timeframe after accounting for real-world trading costs in 2025?**

To answer this, I couldn't just run a simple backtest which would just be overfit and give me unrealistic results. I needed a rigorous test that could re-optimize itself hundreds of times over years of data without cheating (looking ahead). This is known as **Walk-Forward Analysis (WFA)**.

However, WFA is computationally expensive and running thousands of optimization loops takes hours. So, wanting to push my engineering skills, I built a WFA Engine with two different optimization methods: 

1. A Python-only, **Numba**-optimized Engine to reduce overhead and without the complexities of setting up a **C++** Module
2. A **C++** Module which is integrated into the Python Engine for maximum computational speed and high performance.

---

## 2. Methodology (WFA) 
Related Code: (`cpp_wfa.py` and `numba_wfa.py`)

At its core, the system measures how a pair behaves historically and reacts when the spread becomes statistically abnormal. To do this, I used a combination of statistical tests and guards.

### Logic

* **Kalman Filter (Dynamic Hedge Ratio):** Implemented a Kalman Filter to dynamically calculate the hedge ratio ($\beta$) between two assets, allowing the model to adapt instantly to new price information, thus avoiding the lag inherent in simple moving averages.
* **Z-Score:** Measures a spread's deviation from its mean, and this measurement acts as the primary trade signal. The system enters a trade if the magnitude of the current Z-score, $|\mathbf{Z_{current}}|$, exceeds the entry threshold, $\mathbf{Z_{entry}}$.
* **Augmented Dickey-Fuller Test (ADF):** **Stat Test #1** The test is calculated on the preceding In-Sample (IS) window to confirm **stationarity** (a.k.a cointegration). If the **P-value** of the test exceeds a predefined threshold (e.g., $P$-value $> 0.10$), the IS window is deemed non-stationary and the corresponding OOS window is skipped entirely.
* **Hurst Exponent:** **Stat Test #2** Measures the degree of **mean reversion behavior** vs **trending behavior** in the spread. If the Hurst Value exceeds a threshold (e.g., $H > 0.75$), the system detects persistent, trending behavior and blocks the trade at that specific moment.
* **Dollar-Based Stop Loss:** Calculated stops based on **Gross PnL** (real dollars lost before fees), making the optimization more path-dependent and realistic than simple percentage stops.

**Note:** With these components, we heavily reduce trading risk by ensuring that market positions are only initiated when both **statistical confidence (ADF)** and **current behavior (Hurst)** strongly favor mean reversion. The **Dollar-Based Stop Loss** acts as an absolute risk ceiling, protecting capital during black-swan events or rapid mean reversion failures.

### WFA Process 

I implemented a rolling-window approach to eliminate lookahead bias, and the WFA process follows as such:

1.  **In-Sample (Train):** The model trains on **60 days** of data to find the optimal Z-Score thresholds ($Z_{entry}, Z_{exit}, Z_{stop}$). The window is blocked if it exceeds the above mentioned ADF threshold.
2.  **Out-of-Sample (Test):** These parameters are frozen and tested on the next **15 days** of unseen data. Any trades during the OOS are blocked if its Hurst Exponent exceeds the above mentioned Hurst threshold.
3.  **Repeat:** The window slides forward, and the process repeats, mimicking real-world constraints.

---

![System Architecture](assets/wfa_archi.png)
*Figure 1: High-level architecture of the C++ Accelerated Walk-Forward Analysis system.*

---

## 3. Methodology (C++/Numba Comparison) 
Related Code: (`stress.py`)

Initially, my WFA engine was extremely slow—Python for loops were simply not capable of handling the thousands of optimization cycles required for each rolling window. To address this, I first migrated the bottleneck logic into a Numba JIT-compiled function which improved speeds massively. However, while Numba was a major speedup, it still struggled once the project required millions of backtest iterations.

That's when I decided to implement a C++ module to replace run_opt and numba_bt in `numba_wfa` to improve speeds even further.

Hence, the latter part of this project therefore focuses on comparing these two acceleration strategies for the core optimization loop inside the Walk-Forward Analysis (WFA) engine: Numba JIT vs. the C++ module

The main computational bottleneck of WFA occurs in the optimization and backtest loops. For each in-sample window, the engine iterates through hundreds of parameter combinations and executes the backtest logic thousands of times. To evaluate the performance benefit of the C++ module, I made a stress test to compare both implementations.

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

Once the WFA is built and ready to go, the full workflow below lets you run your own walk-forward tests exactly as described in section 2 and figure 1 earlier.

### Fallback Option

If you encounter any issues compiling the C++ module, you can switch to the Python-only Numba engine (slightly slower but still fast):

* Set `wfa_script = "numba_wfa"` in `batch_runner.py`.
* Without the C++ module, you will also not be able to benchmark C++ against Numba in `stress.py`.

---

## 6. Results + Discussion of the Walk-Forward Analysis

### ADF and Hurst

To get a sense of how sensitive the strategy is to stationarity constraints, I ran four core WFA configurations:

* (Hurst 0.8, ADF 0.1)
* (Hurst 0.8, ADF 0.2)
* (Hurst 0.75, ADF 0.1)
* (Hurst 0.75, ADF 0.2)

Then I added two boundary configurations:

* **(Hurst 0.9, ADF 0.2)** — effectively removes most mean-reversion filtering  
* **(Hurst 0.7, ADF 0.2)** — very strict, strongly penalizes trending behavior

These two extremes reveal how the strategy breaks when the filters are either too loose or too restrictive.

The scatter plot below shows each pair’s Walk-Forward out-of-sample performance, with Max Drawdown on the x-axis and Sharpe Ratio on the y-axis. 

Each dot is one full WFA run over four years of 15-minute bars. The size of each dot represents its trade volume, and the colors are arbitrary (alphabetically sorted) and not representative of performance.

![Graphs of Pair Performance (In Sharpe) Across different ADF cutoffs](assets/graphs.png)  
*Figure 2: Compilation of 6 graphs showing Pair Performance in Sharpe against Max Drawdown. (Click for a higher-res view.)*

Across all pairs, several patterns show up consistently:

* **Stricter filters (e.g., 0.75 / 0.1)**  
  * Fewer tradable windows, and while some pairs do improve, it is not a universal effect.

* **Looser filters (e.g., 0.80 / 0.2)**  
  * More trades, but with inflated MDD and no real improvement in Sharpe.

* **A “Goldilocks Zone” emerges (0.75 / 0.2 and 0.80 / 0.1)**  
  * A good middle ground: still decent volume, controlled drawdowns, and stable median Sharpe per pair.

The boundary configs highlight the extremes:

* **(0.9, 0.2)** essentially floods the system with non-stationary spreads.  
  * Result: MDD increases while offering zero upside. The filters are simply too loose.

* **(0.7, 0.2)** is overly strict.  
  * Result: Trade volume collapses, often to near zero — which matches real paper-trading observations where many spreads sit around Hurst 0.75–0.9. Only extremely tight pairs like QQQ/QQQM survive.

Overall, these patterns reinforce why threshold testing matters and why the chosen filters should reflect actual market behavior rather than purely statistical aesthetics. In empirical finance, a Hurst value around 0.7 typically indicates trending (not mean-reverting), and an ADF p-value of 0.2 is far from statistically significant.

---

### Intra-graph Structure

If we zoom in on a typical graph, we can identify four distinct behavioral regions:

![Regions of Pair Performance on 0.8/0.1 Graph](assets/regions.png)  
*Figure 3: Four Distinct Regions (I–IV) on the 0.80 / 0.1 WFA Graph.*

| Region | Observations |
| :--- | :--- |
| **I** (Low MDD, +Sharpe) | Where the genuinely good, alpha-producing pairs reside. |
| **II** (High MDD, ~0 Sharpe) | Wild swings with big wins and losses. This high volatility kills Sharpe. |
| **III** (Low MDD, −Sharpe) | Pairs that aren’t actually cointegrated or very weak. “Mean-reversion” signals are just noise. |
| **IV** (High MDD, −Sharpe) | Hyper-efficient pairs creating high-volume *losing* micro-trades which results in MDD spikes. |

These regions appear consistently across all graphs with enough trade volume.

---

### Top Pairs Found

Across all configurations, the strongest balance appears around **Hurst ≤ 0.8 and ADF ≤ 0.1**, offering:

* a meaningful number of viable trading windows,
* stable out-of-sample Sharpes,
* manageable drawdowns.

This makes **(0.8, 0.1)** the most practical choice for retail execution on 15-minute bars.

The top 25 pairs discovered under this regime (all of which were paper-tested on my [trading algorithm](https://github.com/kaishx)):

| Pairs | Median Sharpe |
| :--- | :--- |
| HD LOW | 0.494 |
| ALL TRV | 0.467 |
| AMAT KLAC | 0.455 |
| SQM ALB | 0.420 |
| ETN PH | 0.415 |
| CBOE CME | 0.344 |
| GEN CHKP | 0.333 |
| AJG ON | 0.309 |
| BX KKR | 0.300 |
| VMC MLM | 0.293 |
| BJ COST | 0.276 |
| BLK TROW | 0.269 |
| DDOG MDB | 0.257 |
| DT DDOG | 0.256 |
| RSG VMC | 0.244 |
| MCD QSR | 0.243 |
| CFG RF | 0.235 |
| MS ORCL | 0.231 |
| NUE STLD | 0.231 |
| ITW MMM | 0.228 |
| DXCM PODD | 0.228 |
| AVGO NVDA | 0.227 |
| CRL IQV | 0.212 |
| DOCS TDOC | 0.211 |
| ALL PGR | 0.208 |

*Figure 4: Top 25 Pairs of the 0.8 / 0.1 Configuration. The Sharpe Values are the Pair's Median Sharpe across 4 WFAs with different starting times.*

---

### Portfolio Sharpe

While individual pair Sharpes are modest (<0.5), diversification across sectors and correlation clusters dramatically boosts the **portfolio-level** Sharpe.

We estimate an upper-bound portfolio Sharpe using:

$$\mathbf{S_{p, \text{upper bound}}} = \frac{\sum_{i=1}^{N} \mathbf{S_i}}{\sqrt{N}}$$

assuming equal volatility across pairs and zero cross-correlation.

| Metric | Value | 
| :--- | :--- | 
| **Number of Pairs ($N$)** | 25 | 
| **Sum of Individual Sharpe Ratios ($\sum S_i$)** | 7.375 |
| **Portfolio Sharpe (Upper Bound, $S_p$)** | **1.475** | 

*Note: The 25 pairs are definitely correlated, and hence the realistic portfolio Sharpe is lower.*

### Summary of Section 6

I found 0.8 / 0.1 to be the most suitable Hurst and ADF thresholds for my WFA, which actually matches up with my observations of pairs on my paper trader. It provides a good balance between stability, performance and robustness in the 15m timeframe. I found the portfolio Sharpe to have an upper bound of 1.475, which is quite decent for Pairs Trading.

---


## 7. Results + Discussion of C++ vs. Numba Performance Benchmark

### Benchmark Results (1,000 runs @ 10,000 bars)

| Module | Mean | Std. Dev | 
| :--- | :--- | :--- |
| Numba | 0.007220s | 0.000734s |
| C++ | 0.005572s | 0.000265s | 

![Image of Benchmark Distribution Graph](assets/benchmark.png)

*Figure 5: Probability Distribution Graphs of C++ (orange) and Numba (blue) across 1000 runs @ 10,000 Bars.*

The benefit of C++ over Numba can be seen and interpreted in two ways:
* **Raw Speed:** C++ eliminated Python overhead during the heavy grid-search loops, leveraging the Zero-Copy technique for superior execution speed. Hence, the C++ module was **~1.30x faster** than the Numba version.
* **Consistency:** The C++ performance distribution is much tighter than the Numba curve. In production, **predictable latency** is crucial, which the C++ engine delivers by being immune to Python's Garbage Collection overhead.

However, we must be aware this benchmark is not perfect and across different batches of 1,000 runs can give a result of C++ being 1.2x - 1.9x faster than Numba. Nevertheless, all batches agree that C++ always has a lower mean and standard deviation compred to Numba.

### Summary of Section 7

We can see that while both are fast, C++ is much more optimized and much better suited for high performance calculations in comparison to Numba, resulting in a 1.30x speedup. It shows why implementing C++ is paramount to improving the overall efficiency of the WFA and workflow.

---

## 8. Discussion & Financial Reality

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

## 9. Limitations & Future Work

* **Execution Lag:** The backtest assumes execution at the exact candle close, which neglects live market execution delays.
* **Single-Core Speed:** The system currently runs on a single thread. Scaling analysis to larger universes requires a multi-threaded C++ implementation.
* **Platform Dependency:** Future work requires porting the custom Windows build system to a Linux/Docker environment for cloud deployment readiness.

---

## 10. Conclusion

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
