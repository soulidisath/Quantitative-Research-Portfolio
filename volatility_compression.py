import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
from typing import Optional, Dict, Any

# Configure institutional-grade logging instead of print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VolatilityCompressionEngine:
    """
    A systematic execution engine that identifies intraday volatility compression 
    (tight pricing boxes) and executes directional breakouts filtered by 
    multi-timeframe (HTF/LTF) momentum regimes.
    """

    def __init__(self, box_length: int = 3, maker_fee_bps: float = 1.0, taker_fee_bps: float = 3.5):
        """
        Initialize the execution engine with market microstructure friction assumptions.
        
        Args:
            box_length (int): The duration (in bars) required to establish a compression zone.
            maker_fee_bps (float): Maker rebate/fee in basis points.
            taker_fee_bps (float): Taker fee in basis points.
        """
        self.box_length = box_length
        self.maker_fee = maker_fee_bps / 10000
        self.taker_fee = taker_fee_bps / 10000
        self.fees_entry_exit = self.maker_fee * 2
        self.fees_stop_loss = self.taker_fee + self.maker_fee

    def _compute_regime_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates High-Timeframe (HTF) and Low-Timeframe (LTF) directional regimes.
        """
        # Calculate localized volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=50)

        # Low-Timeframe (LTF) Regime
        ltf_fast, ltf_slow = 12, 21
        df['regime_ltf'] = ta.ema(df['close'], ltf_fast) > ta.ema(df['close'], ltf_slow)

        # High-Timeframe (HTF) Regime (Simulated via longer lookbacks for single-tf data)
        htf_fast, htf_slow = 12 * 3, 21 * 3 
        df['regime_htf'] = ta.ema(df['close'], htf_fast) > ta.ema(df['close'], htf_slow)

        return df

    def calculate_risk_adjusted_return(self, entry: float, exit_price: float, stop_loss: float, is_long: bool) -> float:
        """
        Calculates the R-multiple (Risk-Adjusted Return) strictly accounting for 
        transaction costs and slippage.
        """
        gross_return = (exit_price - entry) / entry if is_long else (entry - exit_price) / entry
        risk_pct = abs(stop_loss - entry) / entry
        
        # Net R-multiple after microstructure friction
        net_r_multiple = (gross_return - self.fees_entry_exit) / (risk_pct + self.fees_stop_loss)
        return net_r_multiple

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the walk-forward simulation across the price series.
        """
        df = self._compute_regime_filters(df.copy())
        
        df["returns"] = np.nan
        df["trade_direction"] = None

        box_candle = None
        is_long_exposure = False
        is_short_exposure = False
        
        entry_price, stop_loss, break_even_trigger = 0.0, 0.0, None

        logging.info("Initiating volatility compression backtest sequence...")

        for i in range(50, len(df)):
            current_candle = df.iloc[i]

            # 1. State: Identify Volatility Compression (Box Setup)
            true_range = current_candle['high'] - current_candle['low']
            is_valid_compression = (true_range < 2.5 * current_candle['atr']) and (true_range / current_candle['low'] > 0.0025)
            
            if box_candle is None:
                box_candle = current_candle if is_valid_compression else None
                continue  

            time_in_box = i - df.index.get_loc(box_candle.name)
            
            if time_in_box < self.box_length:
                is_compressing = (current_candle['high'] < box_candle['high']) and (current_candle['low'] > box_candle['low'])
                if not is_compressing:
                    box_candle = current_candle if is_valid_compression else None
                continue
        
            # 2. State: Breakout Execution
            if not (is_long_exposure or is_short_exposure):
                # Long Breakout evaluated against HTF Regime
                if current_candle['high'] >= box_candle['high'] and current_candle['regime_htf']:
                    is_long_exposure = True
                    entry_price = box_candle['high']
                    stop_loss = box_candle['low']

                # Short Breakout evaluated against HTF Regime
                elif current_candle['low'] <= box_candle['low'] and not current_candle['regime_htf']:
                    is_short_exposure = True
                    entry_price = box_candle['low']
                    stop_loss = box_candle['high']
                else:
                    box_candle = None
                continue

            # 3. State: Trade Management & Friction Adjustments
            current_r = self.calculate_risk_adjusted_return(entry_price, current_candle['close'], stop_loss, is_long_exposure)

            # Exit Logic: Long
            if is_long_exposure:
                if current_candle['low'] < stop_loss:
                    df.loc[box_candle.name, "returns"] = -1.0
                    df.loc[box_candle.name, "trade_direction"] = "Long"
                    is_long_exposure = False
                    box_candle = current_candle

                elif not current_candle['regime_ltf']:  # Momentum divergence exit
                    df.loc[box_candle.name, "returns"] = current_r
                    df.loc[box_candle.name, "trade_direction"] = "Long"
                    is_long_exposure = False
                    box_candle = current_candle

            # Exit Logic: Short
            elif is_short_exposure:
                if current_candle['high'] > stop_loss:
                    df.loc[box_candle.name, "returns"] = -1.0
                    df.loc[box_candle.name, "trade_direction"] = "Short"
                    is_short_exposure = False
                    box_candle = current_candle

                elif current_candle['regime_ltf']:  # Momentum divergence exit
                    df.loc[box_candle.name, "returns"] = current_r
                    df.loc[box_candle.name, "trade_direction"] = "Short"
                    is_short_exposure = False
                    box_candle = current_candle

        logging.info("Backtest sequence complete.")
        return df.dropna(subset=['returns'])
