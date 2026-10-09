import numpy as np
import pandas as pd
import optuna
import nolds
import warnings
from typing import Dict, Tuple
from dask.distributed import Client
from statsmodels.tsa import stattools
from arch.unitroot import PhillipsPerron
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.simplefilter("ignore")

class MarketRegimeClassifier:
    """
    Evaluates rolling statistical estimators to classify market phases into 
    persistent (trending) or mean-reverting (consolidating) regimes.
    """
    
    def __init__(self, params: Dict[str, float]):
        self.params = params

    @staticmethod
    def _rolling_kpss(close: pd.Series, length: int, threshold: float = 1.0) -> pd.Series:
        def kpss_test(window):
            stat, _, _, _ = stattools.kpss(window, regression='c', nlags="auto")
            return stat
        kpss_values = close.rolling(window=length, min_periods=length).apply(kpss_test)
        return (kpss_values >= threshold).astype(int)

    @staticmethod
    def _rolling_adf(close: pd.Series, length: int, threshold: float = -1.0, smooth: int = 5) -> pd.Series:
        def adf_test(window):
            stat, _, _, _, _, _ = stattools.adfuller(window)
            return stat   
        adf_values = close.rolling(window=length, min_periods=length).apply(adf_test).rolling(smooth).mean()
        return (adf_values >= threshold).astype(int)

    @staticmethod
    def _rolling_hurst(close: pd.Series, length: int, threshold: float = 0.75) -> pd.Series:
        def hurst_test(window):
            return nolds.mfhurst_b(window)
        hurst_exponent = close.rolling(window=length, min_periods=length).apply(hurst_test)
        return (hurst_exponent >= threshold).astype(int)

    @staticmethod
    def _rolling_pp(close: pd.Series, length: int, threshold: float = 0.05, smooth: int = 5) -> pd.Series:
        def pp_test(window):
            return PhillipsPerron(window).stat
        pp_values = close.rolling(window=length, min_periods=length).apply(pp_test).rolling(smooth).mean()
        return (pp_values >= threshold).astype(int)

    def evaluate_regime_score(self, close: pd.Series, mri: pd.Series) -> float:
        """
        Computes the normalized deviation score for mean-reverting vs. trending phases.
        """
        mean_reverting_mask = (mri <= 0.5).astype(int)
        trending_mask = (mri > 0.5).astype(int)

        # Neutral Score Evaluation
        neutral_segments = (mean_reverting_mask != mean_reverting_mask.shift()).cumsum() * mean_reverting_mask
        segment_lengths = neutral_segments.groupby(neutral_segments).transform('size')
        long_neutral_segments = neutral_segments[segment_lengths >= 10]

        segment_means = close.groupby(long_neutral_segments).transform('mean')
        absolute_deviations = np.abs(close - segment_means)
        mean_deviation_per_segment = absolute_deviations.groupby(long_neutral_segments).mean()

        asset_std = close.std()
        neutral_score = 3 * mean_deviation_per_segment.mean() / asset_std if asset_std > 0 else 0
        
        # Trend Score Evaluation
        trend_segments = (trending_mask != trending_mask.shift()).cumsum() * trending_mask
        segment_start_prices = close.groupby(trend_segments).transform('first')
        segment_end_prices = close.groupby(trend_segments).transform('last')
        price_changes = np.abs(segment_end_prices - segment_start_prices)
        
        avg_price_change_per_segment = price_changes.groupby(trend_segments).mean()
        trend_score = avg_price_change_per_segment.mean() / asset_std if asset_std > 0 else 0

        normalized_neutral_score = 1 / (1 + neutral_score)
        normalized_trend_score = 1 / (1 + trend_score)

        return float((normalized_neutral_score + normalized_trend_score) / 2)

    @staticmethod
    def optimize_mri_hyperparameters(data: pd.DataFrame, n_trials: int = 100) -> Tuple[float, Dict[str, float]]:
        """
        Bayesian optimization of the Market Regime Indicator across multiple assets.
        """
        client = Client()
        data_list = [df for _, df in data.groupby(level="coin")]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "adf_length": trial.suggest_int("adf_length", 90, 110),
                "adf_threshold": trial.suggest_float("adf_threshold", -2, -1),
                "kpss_length": trial.suggest_int("kpss_length", 90, 110),
                "kpss_threshold": trial.suggest_float("kpss_threshold", 0.5, 1.5),
                "hurst_length": trial.suggest_int("hurst_length", 60, 80),
                "hurst_threshold": trial.suggest_float("hurst_threshold", 0.48, 0.52),
            }

            def calculate_coin_score(coin_data):
                classifier = MarketRegimeClassifier(params)
                # Compute MRI average logic here...
                mri_dummy = pd.Series(np.random.uniform(0, 1, len(coin_data)), index=coin_data.index)
                return classifier.evaluate_regime_score(coin_data['close'], mri_dummy)

            futures = [client.submit(calculate_coin_score, coin_data) for coin_data in data_list]
            scores = client.gather(futures)
            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        client.close()
        return study.best_value, study.best_params
