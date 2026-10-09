import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModifiedBackwardTimeSelection:
    """
    Implements the modified Backward-in-Time Selection (mBTS) algorithm.
    Utilizes Bayesian Information Criterion (BIC) penalization to dynamically 
    select statistically significant lags across high-dimensional time series.
    """

    def __init__(self, pmax: int):
        self.pmax = pmax
        self.indexV: Optional[np.ndarray] = None
        self.maxorder: Optional[int] = None
        self.coefs: Optional[np.ndarray] = None
        self.varnames: Optional[List[str]] = None

    def _multilagmatrix(self, xM: np.ndarray, responseindex: int, ordersV: np.ndarray, indexV: np.ndarray) -> np.ndarray:
        n, K = xM.shape
        pmax = len(indexV) // K
        
        # Pre-allocate sparse matrix for selected lags
        xtempM = np.full((n, K * pmax), np.nan)
        
        # Populate matrix purely with the lags designated by the orders vector
        for iK in range(K):
            for lag in range(1, ordersV[iK] + 1):
                xtempM[lag:, (iK * pmax) + (lag - 1)] = xM[:-lag, iK]

        # Horizontally stack the response variable with the filtered lag matrix
        xlagM = np.hstack((xM[:, responseindex].reshape(-1, 1), xtempM[:, indexV == 1]))
        
        # Truncate NaNs introduced by the maximum shift
        return xlagM[np.max(ordersV):, :]

    def _dr_fit_mse(self, xM: np.ndarray, responseindex: int, ordersV: np.ndarray, indexV: np.ndarray) -> float:
        xlagM = self._multilagmatrix(xM, responseindex, ordersV, indexV)
        y = xlagM[:, 0].reshape(-1, 1)
        x = xlagM[:, 1:]
        
        # Handle edge case where no lags are selected
        if x.shape[1] == 0:
            return float(np.var(y))
            
        # Utilize Moore-Penrose pseudo-inverse (pinv) instead of standard inverse (inv)
        # to gracefully handle singular matrices in strictly collinear macro datasets
        An = np.linalg.pinv(x.T @ x) @ x.T @ y
        preds = x @ An
        residuals = y - preds
        
        return float(np.mean(residuals.flatten() ** 2))

    def fit(self, df: pd.DataFrame, responseindex: int = 0) -> None:
        xM = df.values.copy()
        n, K = xM.shape
        self.varnames = df.columns.tolist()

        # Mean-normalize variables to center the distribution for OLS evaluation
        for d in range(K):
            xM[:, d] -= np.mean(xM[:, d])

        # Initialize selection vectors
        indexV = np.zeros(K * self.pmax, dtype=int)
        ordersinV = np.zeros(K, dtype=int)

        # Establish baseline MSE and BIC for the zero-lag state
        MSEval = self._dr_fit_mse(xM, responseindex, ordersinV, indexV)
        BICold = (n - np.max(ordersinV)) * np.log(MSEval) + np.sum(ordersinV) * np.log(n - np.max(ordersinV))

        ingameV = np.arange(K)
        ningame = K
        terminateflag = False
        incrisor = 1

        # Core iterative selection loop: evaluates adding lags recursively
        while not terminateflag and ningame != 0:
            # Drop variables that have reached the maximum permitted lag (pmax)
            pmaxreach = np.where(ordersinV >= self.pmax)[0]
            ingameV = np.setdiff1d(ingameV, pmaxreach)
            ningame = len(ingameV)

            if ningame != 0:
                BICnowV = np.full(ningame, np.nan)
                MSEnowV = np.full(ningame, np.nan)

                # Evaluate the marginal improvement of adding the next lag for each variable
                for iK in range(ningame):
                    ordtempV = ordersinV.copy()
                    ordtempV[ingameV[iK]] += incrisor
                    
                    overpmaxV = np.where(ordtempV > self.pmax)[0]
                    if len(overpmaxV) > 0:
                        ordtempV[overpmaxV] = self.pmax
                    if len(overpmaxV) == K:
                        terminateflag = True

                    tempindexV = indexV.copy()
                    tempindexV[(ingameV[iK]) * self.pmax + ordtempV[ingameV[iK]] - 1] = 1
                    
                    MSEnowV[iK] = self._dr_fit_mse(xM, responseindex, ordtempV, tempindexV)
                    
                    # Compute penalized BIC for the temporary matrix
                    BICnowV[iK] = (n - np.max(ordtempV)) * np.log(MSEnowV[iK]) + np.sum(tempindexV) * np.log(n - np.max(ordtempV))

                BICnew = np.min(BICnowV)
                iBICnew = np.argmin(BICnowV)
                invarindex = ingameV[BICnowV == BICnew]

                # If BIC does not improve, increment the lag search depth but do not accept the state
                if BICold <= BICnew:
                    incrisor += 1
                    if incrisor > self.pmax - np.min(ordersinV):
                        terminateflag = True
                # If BIC improves, accept the new lag structure and reset search depth
                else:
                    indexV[(invarindex) * self.pmax + ordersinV[invarindex] + incrisor - 1] = 1
                    ordersinV[invarindex] += incrisor
                    BICold = BICnew
                    incrisor = 1
                    MSEval = MSEnowV[iBICnew]
            else:
                terminateflag = True

        # Finalize the optimal sparse structure
        self.indexV = np.reshape(indexV, (K, self.pmax)).T.reshape(-1)
        self.maxorder = np.max(ordersinV)

    def get_selected_features(self) -> List[str]:
        if self.indexV is None or self.varnames is None:
            raise ValueError("Model must be fitted before extracting features.")
        
        selected = np.where(self.indexV == 1)[0]
        feature_indices = set([idx // self.pmax for idx in selected])
        return [self.varnames[i] for i in feature_indices]


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
        # Evaluate trace statistics against 5% critical values to define cointegration rank
        try:
            johansen_test = coint_johansen(window_data, det_order=0, k_ar_diff=self.max_lag // 2)
            return any(johansen_test.lr1 > johansen_test.cvt[:, 1])
        except np.linalg.LinAlgError:
            return False

    def run_rolling_walk_forward(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        logging.info(f"Initiating rolling walk-forward analysis. Window: {self.window_size}")
        
        # Step linearly through the time series to enforce strict out-of-sample forecasting
        for i in range(self.window_size, len(df) - 1):
            window = df.iloc[i - self.window_size : i]
            target_actual = df.iloc[i][target_col]
            
            # Apply feature selection to drop highly collinear/insignificant macro variables
            if self.apply_mbts:
                mbts = ModifiedBackwardTimeSelection(pmax=self.max_lag)
                mbts.fit(window, responseindex=window.columns.get_loc(target_col))
                selected_cols = mbts.get_selected_features()
                
                # Ensure the dependent variable remains in the dataset
                if target_col not in selected_cols:
                    selected_cols.append(target_col)
                window = window[selected_cols]

            is_cointegrated = self._evaluate_structural_regime(window)
            
            try:
                # Route to VECM if structural equilibrium holds
                if is_cointegrated:
                    best_lag = select_order(window, maxlags=self.max_lag).selected_orders['bic']
                    model = VECM(window, k_ar_diff=best_lag).fit()
                    pred = model.predict(steps=1)[0][window.columns.get_loc(target_col)]
                
                # Route to purely differenced VAR if series are integrated without equilibrium
                else:
                    diff_data = window.diff().dropna()
                    model = VAR(diff_data)
                    best_lag = model.select_order(maxlags=self.max_lag).selected_orders['bic']
                    fitted = model.fit(maxlags=best_lag, verbose=False)
                    diff_pred = fitted.forecast(y=diff_data.values[-best_lag:], steps=1)[0]
                    
                    # Reconstruct the price level from the forecasted first difference
                    last_value = window.iloc[-1][target_col]
                    pred = last_value + diff_pred[window.columns.get_loc(target_col)]
                
                self.forecasts.append(pred)
                self.actuals.append(target_actual)
                self.dates.append(df.index[i])
                
            except Exception:
                # Fallback to naive persistence baseline upon convergence failure
                self.forecasts.append(window.iloc[-1][target_col])
                self.actuals.append(target_actual)
                self.dates.append(df.index[i])

        return pd.DataFrame({'Actual': self.actuals, 'Forecast': self.forecasts}, index=self.dates)


class BenchmarkAndTradingSimulator:
    """
    Evaluates econometric forecasts against the martingale persistence baseline 
    and simulates out-of-sample directional trading filters.
    """
    
    @staticmethod
    def calculate_nrmse_baseline(results_df: pd.DataFrame) -> float:
        actuals = results_df['Actual'].values
        forecasts = results_df['Forecast'].values
        
        # Construct the random walk baseline (Martingale persistence)
        naive_forecasts = np.roll(actuals, shift=1)
        naive_forecasts[0] = actuals[0]
        
        model_rmse = np.sqrt(np.mean((actuals - forecasts) ** 2))
        naive_rmse = np.sqrt(np.mean((actuals - naive_forecasts) ** 2))
        
        # NRMSE >= 1 indicates failure to beat the random walk in point estimation
        nrmse = model_rmse / naive_rmse if naive_rmse > 0 else np.inf
        logging.info(f"Model RMSE: {model_rmse:.4f} | Naive RMSE: {naive_rmse:.4f} | NRMSE: {nrmse:.4f}")
        return float(nrmse)

    @staticmethod
    def simulate_directional_filter(results_df: pd.DataFrame, threshold: float = 0.001) -> Dict[str, float]:
        actuals = results_df['Actual'].values
        forecasts = results_df['Forecast'].values
        
        returns = np.diff(actuals) / actuals[:-1]
        forecast_returns = (forecasts[1:] - actuals[:-1]) / actuals[:-1]
        
        # Generate discrete vectors based on forecast confidence threshold
        signals = np.where(forecast_returns > threshold, 1, np.where(forecast_returns < -threshold, -1, 0))
        
        # Shift signals by 1 strictly to prevent look-ahead bias during execution tracking
        trades = signals[:-1] * returns[1:]
        executed_trades = trades[trades != 0]
        
        if len(executed_trades) == 0:
            return {"Total Trades": 0, "Win Rate": 0.0, "Profit Factor": 0.0}

        winning_trades = executed_trades[executed_trades > 0]
        losing_trades = executed_trades[executed_trades < 0]

        win_rate = len(winning_trades) / len(executed_trades)
        gross_profit = np.sum(winning_trades)
        gross_loss = np.abs(np.sum(losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        metrics = {
            "Total Trades": len(executed_trades),
            "Win Rate": float(np.round(win_rate * 100, 2)),
            "Profit Factor": float(np.round(profit_factor, 2))
        }
        
        logging.info(f"Trading Filter Simulation Results: {metrics}")
        return metrics
