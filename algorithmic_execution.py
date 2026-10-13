import math
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, Tuple, Optional, List
from pybit.unified_trading import HTTP
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketMicrostructureModel:
    """
    Accounts for real-world execution frictions by explicitly modeling 
    maker/taker fee tiers, bid-ask spread slippage, and queue latency.
    """

    def __init__(self, maker_fee_bps: float = 2.0, taker_fee_bps: float = 5.5, slippage_bps: float = 5.0):
        self.maker_fee = maker_fee_bps / 10000
        self.taker_fee = taker_fee_bps / 10000
        self.slippage = slippage_bps / 10000
        
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
            
        if risk_pct <= 0:
            return 0.0

        net_return = gross_return - self.fees_limit_entry_exit
        true_risk = risk_pct + self.fees_market_stop_loss
        
        return float(net_return / true_risk)


class AlgorithmicExecutionClient:
    """
    Asynchronous order-routing wrapper interfacing with Bybit's Unified Trading API.
    Enforces precision sizing, Time-In-Force (TIF) logic, and exchange limits.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.session = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)

    def _get_instrument_precision(self, symbol: str) -> str:
        """Fetches the exchange-defined minimum order quantity step (lot size) as a string."""
        response = self.session.get_instruments_info(category="linear", symbol=symbol)
        tick_size = response["result"]["list"][0]["lotSizeFilter"]["qtyStep"]
        return tick_size

    def route_order(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: Optional[float] = None, limit_price: Optional[float] = None) -> str:
        """
        Routes conditional execution orders to the exchange order book.
        """
        # Fix: Use Decimal module to prevent floating-point precision rejection
        tick_size_str = self._get_instrument_precision(symbol)
        tick_size_dec = Decimal(tick_size_str)
        qty_dec = Decimal(str(qty))
        
        # Exact mathematical flooring without IEEE 754 precision loss
        adjusted_qty_dec = (qty_dec // tick_size_dec) * tick_size_dec
        adjusted_qty = float(adjusted_qty_dec)

        if adjusted_qty <= 0:
            raise ValueError(f"Calculated quantity {qty} falls below minimum tick size {float(tick_size_dec)}")

        position_idx = 1 if side.capitalize() == "Buy" else 2
        order_type = 'Limit' if limit_price else 'Market'
        time_in_force = 'PostOnly' if limit_price else 'ImmediateOrCancel'

        try:
            order_response = self.session.place_order(
                category='linear',
                symbol=symbol,
                side=side.capitalize(),
                orderType=order_type,
                qty=str(adjusted_qty),
                timeInForce=time_in_force,
                positionIdx=position_idx,
                stopLoss=str(stop_loss),
                takeProfit=str(take_profit) if take_profit else None,
                price=str(limit_price) if limit_price else None
            )

            if order_response.get('retCode') != 0:
                raise Exception(f"Exchange rejected order: {order_response.get('retMsg')}")

            order_id = order_response['result']['orderId']
            logging.info(f"Order successfully routed. ID: {order_id} | {side} {adjusted_qty} {symbol}")
            return order_id

        except Exception as e:
            logging.error(f"Order routing failure for {symbol}: {str(e)}")
            raise


class BootstrapValidationEngine:
    # [Rest of your BootstrapValidationEngine code exactly as it was]
    def __init__(self, confidence_level: float = 95.0, iterations: int = 1000):
        self.confidence_level = confidence_level
        self.iterations = iterations

    def _compute_confidence_intervals(self, data_array: np.ndarray) -> Tuple[float, float]:
        lower_percentile = (100 - self.confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        return tuple(np.percentile(data_array, [lower_percentile, upper_percentile]))

    def run_simulation(self, returns: pd.Series) -> Dict[str, Tuple[float, float]]:
        clean_returns = returns.dropna()
        num_trades = len(clean_returns)
        
        if num_trades < 30:
            logging.warning("Sample size < 30. Bootstrap distributions may lack statistical significance.")

        win_rates, kellys, max_drawdowns, evs = [], [], [], []

        for _ in range(self.iterations):
            sampled = clean_returns.sample(n=num_trades, replace=True)
            winning_trades = sampled[sampled > 0]
            losing_trades = sampled[sampled < 0]
            
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            win_rates.append(win_rate)

            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            ev = (win_rate * rr_ratio) - (1 - win_rate)
            evs.append(ev)

            kelly = (win_rate - ((1 - win_rate) / rr_ratio)) if rr_ratio > 0 else 0
            kellys.append(max(0, kelly))  

            cumulative_equity = sampled.cumsum()
            peak = cumulative_equity.cummax()
            drawdown = (peak - cumulative_equity).max() 
            
            theoretical_exposure = max(0.01, kelly / 4) 
            drawdown_percentage = 1 - (1 - theoretical_exposure) ** drawdown
            max_drawdowns.append(drawdown_percentage)

        results = {
            "Expected_Value_95_CI": self._compute_confidence_intervals(np.array(evs)),
            "Quarter_Kelly_Allocation_95_CI": self._compute_confidence_intervals(np.array(kellys) / 4),
            "Max_Drawdown_Limit_95_CI": self._compute_confidence_intervals(np.array(max_drawdowns))
        }

        logging.info("Bootstrap validation complete.")
        for metric, bounds in results.items():
            logging.info(f"{metric}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            
        return results
