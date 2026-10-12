# Quantitative-Research-Portfolio

A Python-based quantitative research and execution environment specializing in macroeconomic time-series forecasting, non-linear market regime classification, and dynamic capital allocation. 

This repository contains the core modeling framework developed alongside my M.Eng. Diploma Thesis at the Aristotle University of Thessaloniki (AUTh), focusing on the transmission dynamics between central bank liquidity flows and high-beta asset valuations.

---

## 1. Macro-Econometric Modeling (`macro_econometric_engine.py`)

To evaluate the causal relationships between macroeconomic liquidity (e.g., TGA, Reverse Repos) and asset prices, this module utilizes rolling-window Vector Error Correction Models (VECM) and Vector Autoregression (VAR).

* **Feature Selection (mBTS):** High-dimensional macroeconomic datasets suffer from severe multicollinearity. This pipeline implements the modified Backward-in-Time Selection (mBTS) algorithm, utilizing Moore-Penrose pseudo-inverses and Bayesian Information Criterion (BIC) penalization to isolate statistically significant lags.
* **Structural Equilibrium Testing:** Dynamically routes data to a VECM if Johansen cointegration trace statistics indicate a structural equilibrium, or to a first-differenced VAR if the series are purely integrated.
* **Martingale Benchmarking:** Out-of-sample directional forecasts are strictly benchmarked against the random walk persistence baseline ($NRMSE \ge 1$).

*(Add a screenshot here showing your VECM forecast plot vs actual price, or the mBTS lag selection matrix)*
![Macro Liquidity Forecast Output](images/vecm_forecast.png)

---

## 2. Systematic Regime & Ensemble Engine (`systematic_ensemble_engine.py`)

Standard trend-following strategies suffer severe drawdowns in mean-reverting environments. This module builds an adaptive, multi-indicator ensemble to dynamically filter trading states.

* **Market Regime Indicator (MRI):** Classifies market states into persistent or mean-reverting phases by evaluating joint significance across Augmented Dickey-Fuller (ADF), KPSS, and rolling Hurst exponent estimators.
* **Bayesian Optimization (Optuna/Dask):** Calibrates ensemble parameters and portfolio weights using Tree-structured Parzen Estimators (TPE). Computations are parallelized across Dask distributed clusters for scale.
* **Curve-Fitting Regularization:** Regularizes the objective function against historical overfitting by generating synthetic Monte Carlo price permutations (permuting inter-bar/intra-bar returns) and applying exponential state-transition decay penalties to penalize high-frequency whipsaws.

*(Screenshot: Ensemble signal generation. The lower panel displays the independent binary states of multiple trend filters. The execution engine (top panel) only allocates capital when the weighted consensus breaches the optimized conviction threshold, effectively filtering false breakouts and minimizing drawdown.)*
<img width="616" height="449" alt="Screenshot 2026-10-12 at 2 54 43 PM" src="https://github.com/user-attachments/assets/835c6471-a655-4eb7-a345-fed6e9a0bfa8" />


*(Add a screenshot here of the MRI identifying trend vs. consolidation, or an Optuna optimization history plot showing convergence)*
![Market Regime Indicator States](images/mri_transitions.png)

---

## 3. Statistical Valuation & Portfolio Optimization (`statistical_valuation.py`)

Financial time-series data severely violates the Gaussian assumptions required for linear statistical modeling. This module enforces distribution normalization and resolves dynamic capital allocation.

* **Distributional Normalization:** Evaluates highly skewed market metrics via Yeo-Johnson, Box-Cox, and power ladder transformations. Automatically selects the optimal mapping by aggregating p-values from Shapiro-Wilk and Kolmogorov-Smirnov goodness-of-fit tests.
* **Constrained SLSQP Allocation:** Solves Modern Portfolio Theory (MPT) mean-variance optimization using Sequential Least Squares Programming (SLSQP). Enforces strict equality constraints ($\sum w_i = 1$) and non-negativity bounds to generate actionable, executable weights.

*(Screenshot: Distributional normalization of fundamental metrics. Highly skewed raw data is transformed into standard normal distributions, verified via Q-Q probability plots. The aggregated statistical deviations are mapped directly onto the asset's price action as a heat map to signal localized overbought/oversold regimes.)*
<img width="1097" height="726" alt="image" src="https://github.com/user-attachments/assets/c74fecaf-ad5e-40c5-9c87-1a5b62542720" />


---

## 4. Algorithmic Execution & Risk Verification (`algorithmic_execution.py`)

Theoretical alpha decays rapidly when exposed to market microstructure friction. This module handles real-world order routing and non-parametric risk verification.

* **Microstructure Frictions:** Explicitly computes true risk-adjusted Expected Value ($EV$) by modeling limit/maker rebates against market/taker fees, order book slippage, and queue latency.
* **Asynchronous Order Routing:** REST API client handling exchange precision formatting (lot size step normalization) and limit vs. market Time-In-Force (TIF) logic (PostOnly vs IOC).
* **Bootstrapping Validation:** 1,000-iteration Monte Carlo bootstrapping engine. Resamples historical trade returns with replacement to compute 95% confidence intervals for Expected Value, Kelly Criterion position sizing, and maximum drawdown limits.

*(Add a screenshot here of your Bootstrap distribution histograms for Expected Value or Max Drawdown)*
![Bootstrap Confidence Intervals](images/bootstrap_distribution.png)

---
**Technology Stack:** `numpy`, `pandas`, `statsmodels`, `scipy`, `optuna`, `dask`, `vectorbt`, `arch`
