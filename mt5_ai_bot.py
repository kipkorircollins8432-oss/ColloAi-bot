
"""
MT5 AUTO TRADING BOT — STARTER VERSION

Strategy: EMA trend + RSI confirmation + ATR stop loss.
Default: ANALYSIS ONLY.
Use a separate MT5 terminal/account for testing.
Live trading is high risk and must be explicitly enabled.
"""

import time
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


# ========== SETTINGS ==========

SYMBOL = "XAUUSD"          # Change to your broker's exact symbol
TIMEFRAME = mt5.TIMEFRAME_M15

LIVE_TRADING = true      #true = analysis ; True = real orders
RISK_PERCENT = 0.5         # Risk target per trade (% of equity)
MAX_POSITIONS = 1
MAX_DAILY_LOSS_PERCENT = 2.0
MAX_SPREAD_POINTS = 100    # Adjust for your broker and symbol
CHECK_SECONDS = 10

ATR_SL_MULTIPLIER = 2.0
REWARD_RISK_RATIO = 2.0

MAGIC = 17102026


# ========== INDICATORS ==========

def get_data():
    rates = mt5.copy_rates_from_pos(
        SYMBOL, TIMEFRAME, 0, 250
    )

    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)

    df["ema_fast"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema_slow"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    change = df["close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=true
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=true
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    previous_close = df["close"].shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    return df


def get_signal(df):
    # Exclude the currently forming candle.
    # Compare two completed candles for a fresh EMA crossover.
    previous = df.iloc[-3]
    candle = df.iloc[-2]

    if pd.isna(candle["rsi"]) or pd.isna(candle["atr"]):
        return None

    crossed_up = (
        previous["ema_fast"] <= previous["ema_slow"]
        and candle["ema_fast"] > candle["ema_slow"]
    )

    crossed_down = (
        previous["ema_fast"] >= previous["ema_slow"]
        and candle["ema_fast"] < candle["ema_slow"]
    )

    if crossed_up and candle["rsi"] >= 55:
        return "BUY"

    if crossed_down and candle["rsi"] <= 45:
        return "SELL"

    return None


# ========== RISK MANAGEMENT ==========

def calculate_volume(order_type, entry, stop):
    account = mt5.account_info()
    info = mt5.symbol_info(SYMBOL)

    if account is None or info is None:
        return 0.0

    risk_cash = account.equity * RISK_PERCENT / 100

    # Estimate the loss at the stop for one lot in account currency.
    pnl_one_lot = mt5.order_calc_profit(
        order_type, SYMBOL, 1.0, entry, stop
    )

    if pnl_one_lot is None or abs(pnl_one_lot) <= 0:
        return 0.0

    raw_volume = risk_cash / abs(pnl_one_lot)

    # Round DOWN to the broker's allowed lot step.
    step = info.volume_step
    volume = (int(raw_volume / step)) * step

    # Never round upward to the broker's minimum lot.
    if volume < info.volume_min:
        return 0.0

    volume = min(volume, info.volume_max)
    volume = round(volume, 8)

    # Verify estimated margin before opening a trade.
    margin = mt5.order_calc_margin(
        order_type, SYMBOL, volume, entry
    )

    if margin is None or margin > account.margin_free * 0.8:
        return 0.0

    return volume


def spread_ok(tick, info):
    if tick is None or info is None or info.point <= 0:
        return False

    spread = (tick.ask - tick.bid) / info.point
    return spread <= MAX_SPREAD_POINTS


# ========== EXECUTION ==========

def open_trade(signal, atr):
    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    account = mt5.account_info()

    if info is None or tick is None or account is None:
        print("Cannot read symbol, price or account.")
        return

    if not spread_ok(tick, info):
        print("Spread too high; skipping.")
        return

    order_type = (
        mt5.ORDER_TYPE_BUY
        if signal == "BUY"
        else mt5.ORDER_TYPE_SELL
    )

    entry = tick.ask if signal == "BUY" else tick.bid

    # Round prices to the symbol's supported digits.
    distance = max(
        atr * ATR_SL_MULTIPLIER,
        info.trade_stops_level * info.point
        + 2 * info.point
    )

    if signal == "BUY":
        sl = round(entry - distance, info.digits)
        tp = round(
            entry + distance * REWARD_RISK_RATIO,
            info.digits
        )
    else:
        sl = round(entry + distance, info.digits)
        tp = round(
            entry - distance * REWARD_RISK_RATIO,
            info.digits
        )

    volume = calculate_volume(order_type, entry, sl)

    if volume <= 0:
        print("Lot too small, margin insufficient or sizing failed.")
        return

    print(
        f"{signal} | {SYMBOL} | volume={volume} "
        f"| entry={entry} | SL={sl} | TP={tp}"
    )

    if not LIVE_TRADING:
        print("ANALYSIS ONLY: no order was sent.")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "EMA_RSI_ATR_BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": info.filling_mode,
    }

    # Ask broker to check the request before sending it.
    check = mt5.order_check(request)

    if check is None:
        print("Order check failed:", mt5.last_error())
        return

    if check.retcode != 0:
        print("Order rejected by pre-check:", check)
        return

    result = mt5.order_send(request)

    if result is None:
        print("Order submission failed:", mt5.last_error())
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        print("ORDER EXECUTED:", result)
    elif result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
        print("PARTIAL EXECUTION — inspect MT5:", result)
    else:
        print("ORDER NOT CONFIRMED:", result)


def main():
    if not mt5.initialize():
        print("MT5 connection failed:", mt5.last_error())
        return

    try:
        account = mt5.account_info()
        info = mt5.symbol_info(SYMBOL)

        if account is None or info is None:
            print("Account or symbol unavailable.")
            return

        if not info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                print("Could not select symbol:", SYMBOL)
                return

        print("Connected to MT5.")
        print("Login:", account.login)
        print("Server:", account.server)
        print("Balance:", account.balance)
        print("Equity:", account.equity)
        print("Live trading:", LIVE_TRADING)

        if LIVE_TRADING and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_REAL:
            print("STOPPED: live mode requires a REAL account.")
            return

        day = datetime.now().date()
        day_start_equity = account.equity
        last_candle_time = None

        while True:
            now = datetime.now()

            if now.date() != day:
                day = now.date()
                current = mt5.account_info()
                if current is None:
                    print("Cannot verify account.")
                    break
                day_start_equity = current.equity

            current = mt5.account_info()
            if current is None:
                print("Account disconnected.")
                break

            # Stop opening new trades after the equity loss limit.
            daily_loss = (
                (day_start_equity - current.equity)
                / max(day_start_equity, 1e-9) * 100
            )

            if daily_loss >= MAX_DAILY_LOSS_PERCENT:
                print("Daily loss limit reached. Bot stopped.")
                break

            df = get_data()

            if df is None:
                print("Waiting for market data...")
                time.sleep(CHECK_SECONDS)
                continue

            # Only act once per newly completed candle.
            candle_time = int(df.iloc[-2]["time"])

            if candle_time == last_candle_time:
                time.sleep(CHECK_SECONDS)
                continue

            last_candle_time = candle_time

            positions = mt5.positions_get(symbol=SYMBOL)

            if positions is None:
                print("Cannot verify open positions.")
                break

            # Avoid adding positions to existing trades on this symbol.
            if len(positions) >= MAX_POSITIONS:
                print("Position limit reached.")
                continue

            signal = get_signal(df)

            if signal:
                open_trade(signal, float(df.iloc[-2]["atr"]))
            else:
                print(now, "No valid signal.")

    except KeyboardInterrupt:
        print("Bot stopped by user.")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
