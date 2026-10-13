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

# [MonteCarloSimulator, MarketRegimeIndicator, and ObjectiveEvaluator remain unchanged...]
# Keep those classes exactly as you had them, simply replacing BayesianEnsembleOptimizer:

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
        
        for ind, weight in zip(self.indicators, weights):
            sig = ind.get_signals(data)[dates]
            composite += weight * sig
            
        return pd.Series(np.sign(composite), index=dates).replace(0, np.nan).ffill()

    # Fix: Accept the Dask client as an argument to prevent recursive process spawning (fork-bombing)
    def optimize_weights(self, data_list: List[pd.DataFrame], dask_client: Client, n_trials: int = 1000):
        dates_list = [df['2018-01-01':].index for df in data_list]
        
        def objective(trial: optuna.Trial) -> float:
            w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(len(self.indicators))])
            w /= np.sum(w)
            
            # Dispatch signal generation to passed Dask workers
            futures = [dask_client.submit(self.generate_composite_signal, data, dates, w) 
                       for data, dates in zip(data_list, dates_list)]
            signals_list = dask_client.gather(futures)
            
            def score_parallel(data, dates, sigs):
                entries = np.where(sigs == 1, True, False)
                exits = np.where(sigs == -1, True, False)
                
                pf = vbt.Portfolio.from_signals(
                    data['close'].loc[dates], 
                    entries=entries, exits=exits, short_entries=exits, short_exits=entries
                )
                # Mocking ObjectiveEvaluator call for context
                return ObjectiveEvaluator.evaluate_score(pf, time_coherence=1.0)

            score_futures = [dask_client.submit(score_parallel, data, dates, sigs) 
                             for sigs, data, dates in zip(signals_list, data_list, dates_list)]
            scores = dask_client.gather(score_futures)
            
            return float(np.mean(scores))
    
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.weights = np.array(list(study.best_params.values()))
        self.weights /= np.sum(self.weights)
        
        # Note: Do not call client.close() here; let the main thread manage cluster lifecycle.
