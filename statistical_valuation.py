import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalValuationEngine:
    """
    Transforms skewed market data into normal distributions to satisfy 
    Gaussian assumptions, and resolves dynamic capital allocation using SLSQP.
    """

    @staticmethod
    def calculate_normality_p_value(data: pd.Series) -> float:
        """
        Aggregates p-values from Shapiro-Wilk, Kolmogorov-Smirnov, and Normaltest.
        """
        shapiro = stats.shapiro(data)[1]
        kstest = stats.kstest(data, 'norm')[1]
        normaltest = stats.normaltest(data)[1]
        return float(np.average([shapiro, kstest, normaltest]))

    @staticmethod
    def test_transformations(data: pd.Series) -> pd.DataFrame:
        """
        Evaluates Box-Cox, Yeo-Johnson, and power ladder transformations to find 
        the optimal normalization mapping.
        """
        # Shift data to support strictly positive requirement for certain transforms
        data_shifted = data + 1 - np.min(data) if np.any(data < 0) else data
        results = []

        log_data = np.log(data_shifted)
        results.append({'Transformation': 'log', 'p-Value': StatisticalValuationEngine.calculate_normality_p_value(log_data)})

        boxcox_data = pd.Series(stats.boxcox(data_shifted)[0])
        results.append({'Transformation': 'boxcox', 'p-Value': StatisticalValuationEngine.calculate_normality_p_value(boxcox_data)})

        yeojohnson_data = pd.Series(stats.yeojohnson(data)[0])
        results.append({'Transformation': 'yeojohnson', 'p-Value': StatisticalValuationEngine.calculate_normality_p_value(yeojohnson_data)})

        for i in np.arange(-2, 2.1, 0.2):
            if np.isclose(i, 0): continue
            power_data = data_shifted ** i
            results.append({'Transformation': f'{i:.2f} power', 'p-Value': StatisticalValuationEngine.calculate_normality_p_value(power_data)})

        df_results = pd.DataFrame(results)
        return df_results.sort_values(by='p-Value', ascending=False)

    @staticmethod
    def constrained_portfolio_optimization(df: pd.DataFrame) -> Dict[str, float]:
        """
        Performs Modern Portfolio Theory (MPT) optimization using Sequential Least 
        Squares Programming (SLSQP) subjected to equality and non-negativity bounds.
        """
        returns = df.pct_change(fill_method=None).dropna()
        mean_returns = returns.mean()
        covariance_matrix = returns.cov()
        num_assets = len(mean_returns)

        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Strict Constraints: weights sum to 1, non-negativity bounds [0, 1]
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1 / num_assets] * num_assets)

        result = minimize(negative_sharpe_ratio,
                          initial_weights,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        if not result.success:
            logging.error(f"Optimization failed: {result.message}")
            raise ValueError(f"Optimization failed: {result.message}")

        optimized_weights = result.x * 100
        optimized_weights[optimized_weights < 1] = 0  # Turnover/friction threshold

        return dict(zip(mean_returns.index, optimized_weights))
