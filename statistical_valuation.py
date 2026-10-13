import logging
import warnings
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StatisticalValuationEngine:
    """
    Transforms skewed market data into normal distributions to satisfy the 
    Gaussian assumptions required for linear statistical modeling, and resolves 
    dynamic capital allocation using constrained optimization (SLSQP).
    """

    @staticmethod
    def calculate_normality_p_value(data: pd.Series) -> float:
        """
        Aggregates p-values from Shapiro-Wilk, Kolmogorov-Smirnov, and D'Agostino's 
        K-squared tests to provide a robust, composite normality score.
        """
        # Drop NaNs to prevent statistical test failures
        clean_data = data.dropna()
        if len(clean_data) < 3:
            return 0.0

        # Shapiro-Wilk: Highly sensitive to deviations in the tails
        shapiro_pval = stats.shapiro(clean_data)[1]
        
        # Kolmogorov-Smirnov: Non-parametric goodness-of-fit against the standard normal
        # Standardize the data first for accurate KS testing
        standardized_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        kstest_pval = stats.kstest(standardized_data, 'norm')[1]
        
        # D'Agostino-Pearson: Tests specifically for skewness and kurtosis deviations
        try:
            normaltest_pval = stats.normaltest(clean_data)[1]
        except ValueError:
            # Fallback if sample size is too small for normaltest
            normaltest_pval = 0.0

        # Return the unweighted average of the three p-values
        return float(np.average([shapiro_pval, kstest_pval, normaltest_pval]))

    @staticmethod
    def test_transformations(data: pd.Series) -> pd.DataFrame:
        """
        Evaluates Box-Cox, Yeo-Johnson, and power ladder transformations to dynamically 
        identify the optimal mapping that maximizes Gaussian normality.
        """
        results = []
        
        # Box-Cox and Log transforms strictly require strictly positive inputs.
        # Shift the distribution if non-positive values exist.
        shift_value = 0.0
        if np.any(data <= 0):
            shift_value = float(np.abs(np.min(data)) + 1e-5)
        
        shifted_data = data + shift_value

        # 1. Logarithmic Transformation
        log_data = np.log(shifted_data)
        results.append({
            'Transformation': 'log', 
            'p-Value': StatisticalValuationEngine.calculate_normality_p_value(log_data)
        })

        # 2. Box-Cox Power Transformation
        try:
            boxcox_data = pd.Series(stats.boxcox(shifted_data)[0])
            results.append({
                'Transformation': 'boxcox', 
                'p-Value': StatisticalValuationEngine.calculate_normality_p_value(boxcox_data)
            })
        except Exception as e:
            logging.debug(f"Box-Cox transformation failed: {e}")

        # 3. Yeo-Johnson Transformation (natively handles negative values)
        try:
            yeojohnson_data = pd.Series(stats.yeojohnson(data.dropna())[0])
            results.append({
                'Transformation': 'yeojohnson', 
                'p-Value': StatisticalValuationEngine.calculate_normality_p_value(yeojohnson_data)
            })
        except Exception as e:
            logging.debug(f"Yeo-Johnson transformation failed: {e}")

        # 4. Power Ladder Transformations (Squares, Cubes, Square Roots, Inverses)
        for i in np.arange(-2, 2.1, 0.2):
            if np.isclose(i, 0): 
                continue  # Zero power is equivalent to Log, already computed
                
            power_data = shifted_data ** i
            results.append({
                'Transformation': f'{i:.2f} power', 
                'p-Value': StatisticalValuationEngine.calculate_normality_p_value(power_data)
            })

        df_results = pd.DataFrame(results)
        
        # Sort descending to bring the transformation with the highest normality (p-value) to the top
        return df_results.sort_values(by='p-Value', ascending=False).reset_index(drop=True)

    @staticmethod
    def constrained_portfolio_optimization(returns_df: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Solves the Modern Portfolio Theory (MPT) mean-variance optimization problem.
        Utilizes the Sequential Least Squares Programming (SLSQP) algorithm to enforce 
        strict equality constraints (capital allocation = 100%) and bounds.
        """
        returns = returns_df.dropna()
        mean_returns = returns.mean()
        covariance_matrix = returns.cov()
        num_assets = len(mean_returns)

        # Objective Function: We minimize the negative Sharpe ratio to maximize the actual Sharpe
        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Penalize zero-volatility to prevent division-by-zero errors
            if portfolio_volatility == 0:
                return 0.0
                
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe

        # Equality Constraint: The sum of all asset weights must exactly equal 1 (100% allocation)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})
        
        # Non-Negativity Bounds: Weights must be between 0 and 1 (no shorting, no leverage)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        # Initial Guess: Equal weight distribution across all assets
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # Execute SLSQP solver
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logging.error(f"SLSQP Optimization failed to converge: {result.message}")
            raise ValueError(f"Optimization failed: {result.message}")

        # Convert fractional weights to percentage allocations
        optimized_weights = result.x * 100
        
        # Apply a practical turnover limit: zero out allocations below 1% to save on execution friction
        optimized_weights[optimized_weights < 1.0] = 0.0
        
        # Re-normalize weights to exactly 100% after zeroing out fractional dust
        optimized_weights = (optimized_weights / np.sum(optimized_weights)) * 100

        return dict(zip(mean_returns.index, optimized_weights))
