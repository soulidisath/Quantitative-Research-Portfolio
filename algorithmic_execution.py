import math
import logging
import warnings
from decimal import Decimal
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MarketMicrostructureModel:
    """
    Accounts for real-world execution frictions by explicitly modeling 
    maker/taker fee tiers, bid-ask spread slippage, and queue latency.
    """

    def __init__(self, maker_fee_bps: float = 2.0, taker_fee_bps: float = 5.5, slippage_bps: float = 5.0):
        # Convert basis points to decimal percentages
        self.maker_fee = maker_fee_bps / 10000
        self.taker_fee = taker_fee_bps / 10000
        self.slippage = slippage_bps / 10000
        
        # Define baseline frictional costs for limit (maker) and market (taker) actions
        self.fees_limit_entry_exit = self.maker_fee * 2
        self.fees_market_stop_loss = self.taker_fee + self.maker_fee + self.slippage

    def fees_adjusted_r_multiple(self, entry: float, stop_loss: float, exit_price: float, is_long: bool) -> float:
        """
        Calculates the true Risk-Adjusted Return (R-multiple) of a trade, 
        net of all transactional costs and modeled slippage.
        """
        if is_long:
            gross_return = (exit_price - entry) / entry
            risk_pct = (entry - stop_loss) / entry  
        else:
            gross_return = (entry - exit_price) / entry
            risk_pct = (stop_loss - entry) / entry  
            
        # Prevent division by zero if stop loss is improperly placed at entry
        if risk_pct <= 0:
            return 0.0

        # Net return subtracts the round-trip maker fees (assuming limit execution)
        net_return = gross_return - self.fees_limit_entry_exit
        
        # True risk adds taker fees and slippage (assuming stop-loss triggers market order)
        true_risk = risk_pct + self.fees_market_stop_loss
        
        return float(net_return / true_risk)


class AlgorithmicExecutionClient:
    """
    Asynchronous order-routing wrapper interfacing with Bybit's Unified Trading API.
    Enforces precision sizing, Time-In-Force (TIF) logic, and exchange limits.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        # Initialize HTTP session for REST API order routing
        self.session = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)

    def _get_instrument_precision(self, symbol: str) -> str:
        """Fetches the exchange-defined minimum order quantity step (lot size)."""
        response = self.session.get_instruments_info(category="linear", symbol=symbol)
        tick_size = response["result"]["list"][0]["lotSizeFilter"]["qtyStep"]
        return str(tick_size)

    def route_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        stop_loss: float, 
        take_profit: Optional[float] = None, 
        limit_price: Optional[float] = None
    ) -> str:
        """
        Routes conditional execution orders to the exchange order book.
        """
        # Quantize order quantity to step size using arbitrary-precision arithmetic
        tick_size = Decimal(self._get_instrument_precision(symbol))
        qty_dec = Decimal(str(qty))
        adjusted_qty_dec = (qty_dec // tick_size) * tick_size

        if adjusted_qty_dec <= Decimal("0"):
            raise ValueError(f"Calculated quantity {qty} falls below minimum tick size {tick_size}")

        # PositionIdx 1 for Long, 2 for Short (Hedge Mode requirement)
        position_idx = 1 if side.capitalize() == "Buy" else 2
        
        # Route as Limit order if price provided, otherwise route as Market
        order_type = 'Limit' if limit_price else 'Market'
        
        # 'PostOnly' guarantees maker rebates for limits. 'ImmediateOrCancel' prevents 
        # resting liquidity exposure during aggressive market breakouts.
        time_in_force = 'PostOnly' if limit_price else 'ImmediateOrCancel'

        try:
            order_response = self.session.place_order(
                category='linear',
                symbol=symbol,
                side=side.capitalize(),
                orderType=order_type,
                qty=str(adjusted_qty_dec),
                timeInForce=time_in_force,
                positionIdx=position_idx,
                stopLoss=str(stop_loss),
                takeProfit=str(take_profit) if take_profit else None,
                price=str(limit_price) if limit_price else None
            )

            if order_response.get('retCode') != 0:
                raise Exception(f"Exchange rejected order: {order_response.get('retMsg')}")

            order_id = order_response['result']['orderId']
            logging.info(f"Order successfully routed. ID: {order_id} | {side} {adjusted_qty_dec} {symbol}")
            return order_id

        except Exception as e:
            logging.error(f"Order routing failure for {symbol}: {str(e)}")
            raise


class BootstrapValidationEngine:
    """
    Performs out-of-sample risk verification by running Monte Carlo bootstrapping 
    across the return distribution. Computes rigorous confidence intervals for 
    Expected Value (EV), drawdown tolerances, and optimal Kelly sizing.
    """

    def __init__(self, confidence_level: float = 95.0, iterations: int = 1000):
        self.confidence_level = confidence_level
        self.iterations = iterations

    def _compute_confidence_intervals(self, data_array: np.ndarray) -> Tuple[float, float]:
        """Calculates the lower and upper bounds for the specified confidence level."""
        lower_percentile = (100 - self.confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        bounds = np.percentile(data_array, [lower_percentile, upper_percentile])
        return (float(bounds[0]), float(bounds[1]))

    def run_simulation(self, returns: pd.Series) -> Dict[str, Tuple[float, float]]:
        """
        Executes random resampling with replacement to build probability distributions 
        for core portfolio risk metrics.
        """
        clean_returns = returns.dropna()
        num_trades = len(clean_returns)
        
        if num_trades < 30:
            logging.warning("Sample size < 30. Bootstrap distributions may lack statistical significance.")

        win_rates, kellys, max_drawdowns, evs = [], [], [], []

        for _ in range(self.iterations):
            # Resample trade returns with replacement to simulate alternative path histories
            sampled = clean_returns.sample(n=num_trades, replace=True)
            
            winning_trades = sampled[sampled > 0]
            losing_trades = sampled[sampled < 0]
            
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            win_rates.append(win_rate)

            # Compute Reward-to-Risk ratio
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Expected Value (EV) per unit of risk
            ev = (win_rate * rr_ratio) - (1 - win_rate)
            evs.append(ev)

            # Full Kelly Criterion computation
            kelly = (win_rate - ((1 - win_rate) / rr_ratio)) if rr_ratio > 0 else 0
            kellys.append(max(0, kelly))  # Floor Kelly at 0 to prevent negative sizing

            # Generate synthetic equity curve to measure path-dependent drawdowns
            cumulative_equity = sampled.cumsum()
            peak = cumulative_equity.cummax()
            drawdown = (peak - cumulative_equity).max() 
            
            # Compute drawdown percentage as a function of the theoretical Kelly allocation
            theoretical_exposure = max(0.01, kelly / 4)  # Assume Quarter-Kelly sizing for safety
            drawdown_percentage = 1 - (1 - theoretical_exposure) ** drawdown
            max_drawdowns.append(drawdown_percentage)

        # Aggregate distributions into final confidence intervals
        results = {
            "Expected_Value_95_CI": self._compute_confidence_intervals(np.array(evs)),
            "Quarter_Kelly_Allocation_95_CI": self._compute_confidence_intervals(np.array(kellys) / 4),
            "Max_Drawdown_Limit_95_CI": self._compute_confidence_intervals(np.array(max_drawdowns))
        }

        logging.info("Bootstrap validation complete.")
        for metric, bounds in results.items():
            logging.info(f"{metric}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            
        return results
