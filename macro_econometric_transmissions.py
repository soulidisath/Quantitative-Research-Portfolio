import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from numpy.linalg import LinAlgError
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# [ModifiedBackwardTimeSelection remains unchanged...]

class RollingMacroEconometricEngine:
    """
    Walk-forward econometric research pipeline.
    Evaluates dynamic transmission channels between macroeconomic liquidity flows 
    and asset valuations using rolling windows, adaptive lag selection (mBTS), 
    and dynamic regime switching (VAR vs VECM).
    """

    def __init__(self, window_size: int = 252, max_lag: int = 15, apply_mbts: bool = True):
        self.window_size = window_size
        self.max_lag = max_lag
        self.apply_mbts = apply_mbts
        self.forecasts = []
        self.actuals = []
        self.dates = []

    def _evaluate_structural_regime(self, window_data: pd.DataFrame) -> bool:
        try:
            johansen_test = coint_johansen(window_data, det_order=0, k_ar_diff=self.max_lag // 2)
            return any(johansen_test.lr1 > johansen_test.cvt[:, 1])
        except LinAlgError:
            return False

    def run_rolling_walk_forward(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logging.info(f"Initiating rolling walk-forward analysis. Window: {self.window_size}")
        
        for i in range(self.window_size, len(df) - 1):
            window = df.iloc[i - self.window_size : i]
            target_actual = df.iloc[i][target_col]
            
            if self.apply_mbts:
                mbts = ModifiedBackwardTimeSelection(pmax=self.max_lag)
                mbts.fit(window, responseindex=window.columns.get_loc(target_col))
                selected_cols = mbts.get_selected_features()
                
                if target_col not in selected_cols:
                    selected_cols.append(target_col)
                window = window[selected_cols]

            is_cointegrated = self._evaluate_structural_regime(window)
            
            try:
                if is_cointegrated:
                    best_lag = select_order(window, maxlags=self.max_lag).selected_orders['bic']
                    model = VECM(window, k_ar_diff=best_lag).fit()
                    pred = model.predict(steps=1)[0][window.columns.get_loc(target_col)]
                
                else:
                    diff_data = window.diff().dropna()
                    model = VAR(diff_data)
                    best_lag = model.select_order(maxlags=self.max_lag).selected_orders['bic']
                    fitted = model.fit(maxlags=best_lag, verbose=False)
                    diff_pred = fitted.forecast(y=diff_data.values[-best_lag:], steps=1)[0]
                    
                    last_value = window.iloc[-1][target_col]
                    pred = last_value + diff_pred[window.columns.get_loc(target_col)]
                
                self.forecasts.append(pred)
                self.actuals.append(target_actual)
                self.dates.append(df.index[i])
                
            # Fix: Explicitly catch LinAlgError from statsmodels attempting to invert singular/collinear matrices
            except (LinAlgError, ValueError) as e:
                logging.debug(f"Matrix inversion failure at window {i}: {str(e)}. Falling back to persistence baseline.")
                self.forecasts.append(window.iloc[-1][target_col])
                self.actuals.append(target_actual)
                self.dates.append(df.index[i])

        return pd.DataFrame({'Actual': self.actuals, 'Forecast': self.forecasts}, index=self.dates)

# [BenchmarkAndTradingSimulator remains unchanged...]
