import time
import numpy as np
import pandas as pd
import vectorbt as vbt
import optuna
import nolds
from dask.distributed import Client
from typing import List, Tuple, Optional, Dict
from statsmodels.tsa import stattools
from arch.unitroot import PhillipsPerron
import warnings

warnings.filterwarnings("ignore")

class MonteCarloSimulator:
    """
    Generates synthetic price paths using Monte Carlo permutations of inter-bar 
    and intra-bar log returns. This regularizes the objective function against 
    curve-fitting by testing the strategy on structurally identical but temporally 
    permuted price distributions.
    """
    
    @staticmethod
    def generate_paths(ohlcv: pd.DataFrame, n_simulations: int = 10) -> List[pd.DataFrame]:
        paths = []
        
        # Convert to log returns to ensure additive properties during path reconstruction
        log_data = np.log(ohlcv.astype("float32"))
        
        # Extract intra-bar variances (High, Low, Close relative to Open)
        delta_h = log_data["high"] - log_data["open"]
        delta_l = log_data["low"] - log_data["open"]
        delta_c = log_data["close"] - log_data["open"]
        
        # Extract inter-bar gaps (Close to next Open)
        inter_otc = log_data["open"].iloc[1:].values - log_data["close"].iloc[:-1].values

        for _ in range(n_simulations):
            # Permute inter-bar sequences to break serial correlation while preserving distribution shape
            idx_inter = np.random.permutation(len(ohlcv) - 1)
            # Permute intra-bar sequences to test robustness against localized volatility clustering
            idx_intra = np.random.permutation(len(ohlcv) - 2)
            
            diff_h = np.concatenate((delta_h.iloc[1:-1].values[idx_intra], [delta_h.iloc[-1]]))
            diff_l = np.concatenate((delta_l.iloc[1:-1].values[idx_intra], [delta_l.iloc[-1]])) 
            diff_c = np.concatenate((delta_c.iloc[1:-1].values[idx_intra], [delta_c.iloc[-1]]))
            diff_inter = inter_otc[idx_inter]
            
            # Reconstruct the synthetic price path starting from the actual initial price
            new_o, new_h, new_l, new_c = [log_data["open"].iloc[0]], [log_data["high"].iloc[0]], [log_data["low"].iloc[0]], [log_data["close"].iloc[0]]
            last_close = new_c[0]
            
            for dh, dl, dc, inter in zip(diff_h, diff_l, diff_c, diff_inter):
                o = last_close + inter
                new_o.append(o)
                new_h.append(o + dh)
                new_l.append(o + dl)
                new_c.append(o + dc)
                last_close = o + dc
            
            synth_df = pd.DataFrame({
                "open": new_o,
                "high": new_h,
                "low": new_l,
                "close": new_c,
            })
            
            # Exponentiate back to standard price space and append
            paths.append(np.exp(synth_df).set_index(ohlcv.index))
            
        return paths


class MarketRegimeIndicator:
    """
    Classifies market states into persistent (trending) or mean-reverting (consolidating) 
    regimes by evaluating joint significance across stationarity tests and fractal dimensions.
    """

    @staticmethod
    def compute_regime_matrix(close: pd.Series, lookback: int = 100) -> pd.DataFrame:
        
        # Augmented Dickey-Fuller (ADF): Tests for unit root (non-stationarity)
        def adf_test(window):
            return stattools.adfuller(window)[0]
            
        # KPSS: Null hypothesis is stationarity (opposite of ADF)
        def kpss_test(window):
            return stattools.kpss(window, regression='c', nlags="auto")[0]
            
        # Hurst Exponent: Evaluates long-term memory (H > 0.5 indicates trending behavior)
        def hurst_test(window):
            return nolds.mfhurst_b(window)

        # Apply rolling evaluations. Use min_periods to allow early convergence.
        adf_stat = close.rolling(window=lookback, min_periods=lookback).apply(adf_test)
        kpss_stat = close.rolling(window=lookback, min_periods=lookback).apply(kpss_test)
        hurst_stat = close.rolling(window=lookback, min_periods=lookback).apply(hurst_test)

        # Binarize signals based on empirical critical thresholds
        regime_df = pd.DataFrame({
            'ADF_Trend': (adf_stat >= -1.0).astype(int),
            'KPSS_Trend': (kpss_stat >= 1.0).astype(int),
            'Hurst_Trend': (hurst_stat >= 0.75).astype(int)
        }, index=close.index)
        
        # Calculate composite probability of a trending regime [0.0 to 1.0]
        regime_df['Composite_Trend_Prob'] = regime_df.mean(axis=1)
        return regime_df


class ObjectiveEvaluator:
    """
    Evaluates risk-adjusted returns subject to state-transition coherence penalties.
    """
    
    @staticmethod
    def compute_time_coherence(signals: np.ndarray, composite_signal: np.ndarray) -> float:
        """
        Applies a state-transition exponential decay penalty to mitigate high-frequency 
        whipsaws. Penalizes strategies that rely on clustered, rapid regime shifts.
        """
        signals = np.nan_to_num(signals, 0)
        penalties = []
        
        for signal in signals:
            trade_indices = np.where(np.diff(signal) != 0)[0]
            if len(trade_indices) == 0:
                penalties.append(-1)
                continue
            
            # Compute bar distance between transitions
            trade_distances = np.diff(trade_indices)
            
            # Exponentially penalize transitions occurring within 10 bars of each other
            decay_penalty = np.exp(-0.3 * (trade_distances - 10))
            coherence = 1 - (np.sum(decay_penalty) / len(trade_indices))
            penalties.append(coherence)

        trend_coherence = np.mean(signals == composite_signal)
        std_trades = np.std(np.sum((np.diff(signals, axis=1) != 0), axis=1))
        
        # Penalize variance in trade counts across different constituent indicators
        trades_coherence = 1 - (0.05 * std_trades)

        return float((np.mean(penalties) + trend_coherence + trades_coherence) / 3)

    @staticmethod
    def evaluate_score(pf: vbt.Portfolio, time_coherence: float) -> float:
        """
        Custom objective function blending Sharpe, Sortino, and Omega ratios, heavily 
        penalized by Maximum Drawdown constraints and time coherence.
        """
        stats = pf.stats()
        worst_trade = min(stats.get('Worst Trade [%]', 0), 0)
        max_dd = pf.max_drawdown() * 100
        
        # Non-linear drawdown penalty to aggressively reject strategies with severe tail risk
        dd_penalty = -25 / ((max_dd + worst_trade) / 2) if (max_dd + worst_trade) != 0 else 0
        
        # Aggregate multiple risk-adjusted return ratios to smooth distributional bias
        ratios_avg = (pf.sharpe_ratio() / 2 + pf.sortino_ratio() / 2.9 + pf.omega_ratio() / 1.31) / 3

        profit_factor = min(stats.get('Profit Factor', 0), 10)
        win_rate = stats.get('Win Rate [%]', 0) / 100
        win_factor_avg = (profit_factor / 4 + win_rate / 0.65) / 2 
        
        base_score = (dd_penalty + ratios_avg + win_factor_avg) / 3
        
        # Apply the coherence penalty as an exponential scaler
        coherence_factor = np.exp((time_coherence - 1))
        
        return min(base_score * coherence_factor, 2.0)


class BayesianEnsembleOptimizer:
    """
    Synthesizes non-linear indicators and deploys Bayesian optimization (TPE) 
    via distributed Dask clusters to calibrate equality-constrained portfolio weights.
    """
    
    def __init__(self, indicators: List[object]):
        self.indicators = indicators
        self.weights = np.ones(len(indicators)) / len(indicators)

    def generate_composite_signal(self, data: pd.DataFrame, dates: pd.Index = None, weights: Optional[np.ndarray] = None) -> pd.Series:
        if dates is None:
            dates = data.index
        if weights is None:
            weights = self.weights
            
        composite = np.zeros(len(data.loc[dates]))
        
        # Compute linearly weighted vector based on component allocations
        for ind, weight in zip(self.indicators, weights):
            sig = ind.get_signals(data)[dates]
            composite += weight * sig
            
        # Extract directional state from the composite vector
        return pd.Series(np.sign(composite), index=dates).replace(0, np.nan).ffill()

    def optimize_weights(self, data_list: List[pd.DataFrame], n_trials: int = 1000):
        client = Client()
        dates_list = [df['2018-01-01':].index for df in data_list]
        
        def objective(trial: optuna.Trial) -> float:
            # Sample weights from continuous search space
            w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(len(self.indicators))])
            
            # Enforce equality constraint (weights must sum to 1.0)
            w /= np.sum(w)
            
            # Dispatch signal generation to Dask workers
            futures = [client.submit(self.generate_composite_signal, data, dates, w) 
                       for data, dates in zip(data_list, dates_list)]
            signals_list = client.gather(futures)
            
            # Internal function for distributed backtesting
            def score_parallel(data, dates, sigs):
                entries = np.where(sigs == 1, True, False)
                exits = np.where(sigs == -1, True, False)
                
                pf = vbt.Portfolio.from_signals(
                    data['close'].loc[dates], 
                    entries=entries, exits=exits, short_entries=exits, short_exits=entries
                )
                # Compute objective score subject to coherence penalties
                return ObjectiveEvaluator.evaluate_score(pf, time_coherence=1.0)

            # Dispatch backtests to Dask workers
            score_futures = [client.submit(score_parallel, data, dates, sigs) 
                             for sigs, data, dates in zip(signals_list, data_list, dates_list)]
            scores = client.gather(score_futures)
            
            # Target maximization of average multi-asset objective score
            return float(np.mean(scores))
    
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Apply optimal bounds
        self.weights = np.array(list(study.best_params.values()))
        self.weights /= np.sum(self.weights)
        
        client.close()
