
"""
ColloAI-bot — XAUUSD Gold Trading Engine

Requires:
- Windows computer or Windows VPS
- MetaTrader 5 desktop terminal installed and logged in
- Python packages: MetaTrader5, pandas

Safety: LIVE_TRADING is False by default.
This is a rule-based starter, NOT a verified AI or profitable strategy.
"""

import json
import time
from datetime import datetime, time as datetime_time
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


# ========== SETTINGS ==========

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

LIVE_TRADING = False  # Keep False until thoroughly tested

BOT_MAGIC = 17102026
RISK_PERCENT = 0.25
MAX_OPEN_POSITIONS = 2
MAX_DAILY_LOSS_PERCENT = 1.0
MIN_FREE_MARGIN_PERCENT = 50.0
MAX_SPREAD_POINTS = 100

CHECK_EVERY_SECONDS = 15
DEVIATION_POINTS = 30

# Stores the daily equity baseline on the computer running this bot.
STATE_FILE = Path("colloai_state.json")


# ========== DAILY RISK CONTROL ==========

def load_daily_baseline():
    today = datetime.now().date().isoformat()

    try:
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
            if state.get("date") == today:
                return float(state["starting_equity"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    account = mt5.account_info()
    if account is None:
        raise RuntimeError("Cannot read trading account information.")

    baseline = float(account.equity)
    STATE_FILE.write_text(json.dumps({
        "date": today,
        "starting_equity": baseline
    }))
    return baseline


def daily_loss_limit_hit(starting_equity):
    account = mt5.account_info()
    if account is None or starting_equity <= 0:
        return True

    loss_percent = (
        (starting_equity - account.equity) / starting_equity
    ) * 100

    if loss_percent >= MAX_DAILY_LOSS_PERCENT:
        print(f"DAILY LOSS LIMIT reached: {loss_percent:.2f}%")
        return True

    return False


# ========== MT5 CONNECTION ==========

def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise RuntimeError(
            "No MT5 account connected. Open the MT5 desktop terminal "
            "and log in first."
        )

    print("MT5 connection established.")
    print(f"Account login: {account.login}")
    print(f"Server: {account.server}")
    print(f"Balance: {account.balance:.2f}")
    print(f"Equity: {account.equity:.2f}")
    print(f"Live order submission enabled: {LIVE_TRADING}")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(
            f"Cannot select {SYMBOL}. Your broker may use a name such as "
            "XAUUSD.a or GOLD. Check Market Watch and change SYMBOL."
        )


# ========== MARKET DATA AND SIGNALS ==========

def get_signal():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)

    if rates is None or len(rates) < 60:
        print("Not enough market data.")
        return None

    df = pd.DataFrame(rates)

    # Ignore the still-forming candle.
    df = df.iloc[:-1].copy()

    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()

    change = df["close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    previous = df.iloc[-2]
    candle = df.iloc[-1]

    # Only act on a newly closed candle.
    candle_time = int(candle["time"])

    if candle_time == getattr(get_signal, "last_candle", None):
        return None

    get_signal.last_candle = candle_time

    if pd.isna(candle["rsi"]) or pd.isna(candle["atr"]):
        return None

    print(
        f"Gold close={candle['close']:.2f}, "
        f"EMA20={candle['ema_fast']:.2f}, "
        f"EMA50={candle['ema_slow']:.2f}, "
        f"RSI={candle['rsi']:.1f}, ATR={candle['atr']:.2f}"
    )

    crossed_up = (
        previous["ema_fast"] <= previous["ema_slow"]
        and candle["ema_fast"] > candle["ema_slow"]
    )
    crossed_down = (
        previous["ema_fast"] >= previous["ema_slow"]
        and candle["ema_fast"] < candle["ema_slow"]
    )

    if crossed_up and 50 <= candle["rsi"] <= 70:
        return {"direction": "BUY", "atr": float(candle["atr"])}

    if crossed_down and 30 <= candle["rsi"] <= 50:
        return {"direction": "SELL", "atr": float(candle["atr"])}

    return None


# ========== POSITION AND VOLUME CHECKS ==========

def open_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        raise RuntimeError(f"Cannot read positions: {mt5.last_error()}")
    return list(positions)


def calculate_volume(stop_distance):
    account = mt5.account_info()
    info = mt5.symbol_info(SYMBOL)

    if account is None or info is None or stop_distance <= 0:
        return 0.0

    risk_money = account.equity * (RISK_PERCENT / 100.0)

    # Estimate loss for one lot using the symbol's tick values.
    if info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
        print("Invalid tick size/value; cannot safely calculate volume.")
        return 0.0

    loss_per_lot = (
        stop_distance / info.trade_tick_size
    ) * info.trade_tick_value

    if loss_per_lot <= 0:
        return 0.0

    raw_volume = risk_money / loss_per_lot

    # Round DOWN to broker-supported volume steps.
    step = info.volume_step
    if step <= 0:
        return 0.0

    volume = (raw_volume // step) * step
    volume = min(volume, info.volume_max)

    if volume < info.volume_min:
        print(
            "Calculated volume is below the broker minimum. "
            "No trade will be placed."
        )
        return 0.0

    decimals = max(0, len(str(step).split(".")[-1].rstrip("0")))
    return round(volume, decimals)


def enough_margin():
    account = mt5.account_info()

    if account is None or account.equity <= 0:
        return False

    free_margin_percent = account.margin_free / account.equity * 100

    if free_margin_percent < MIN_FREE_MARGIN_PERCENT:
        print(f"Free margin too low: {free_margin_percent:.1f}%")
        return False

    return True


# ========== ORDER SUBMISSION ==========

def submit_trade(signal):
    if not LIVE_TRADING:
        print(
            f"Signal only: {signal['direction']} {SYMBOL}. "
            "No real order sent because LIVE_TRADING=False."
        )
        return

    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)

    if info is None or tick is None:
        print("Cannot obtain current symbol price.")
        return

    if not info.visible and not mt5.symbol_select(SYMBOL, True):
        print("Cannot make symbol visible.")
        return

    spread = (tick.ask - tick.bid) / info.point
    if spread > MAX_SPREAD_POINTS:
        print(f"Spread too high: {spread:.1f} points. Trade skipped.")
        return

    if not enough_margin():
        return

    positions = open_positions()
    if len(positions) >= MAX_OPEN_POSITIONS:
        print("Maximum open XAUUSD positions reached.")
        return

    direction = signal["direction"]
    is_buy = direction == "BUY"

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    entry = tick.ask if is_buy else tick.bid

    stop_distance = signal["atr"] * 2.0
    target_distance = signal["atr"] * 3.0

    # Respect broker's minimum stop distance.
    min_stop = info.trade_stops_level * info.point
    stop_distance = max(stop_distance, min_stop + 2 * info.point)
    target_distance = max(target_distance, min_stop + 2 * info.point)

    if is_buy:
        sl = entry - stop_distance
        tp = entry + target_distance
    else:
        sl = entry + stop_distance
        tp = entry - target_distance

    sl = round(sl, info.digits)
    tp = round(tp, info.digits)

    volume = calculate_volume(abs(entry - sl))
    if volume <= 0:
        return

    request_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION_POINTS,
        "magic": BOT_MAGIC,
        "comment": "ColloAI-Gold",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Try broker-supported fill policies in turn.
    filling_options = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN,
    ]

    for filling in filling_options:
        request = dict(request_base)
        request["type_filling"] = filling

        check = mt5.order_check(request)
        if check is None:
            print(f"Order check failed: {mt5.last_error()}")
            continue

        if check.retcode != 0:
            print(f"Order rejected by pre-check: {check.comment}")
            continue

        result = mt5.order_send(request)

        if result is None:
            print(f"Order send failed: {mt5.last_error()}")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(
                f"ORDER FILLED: {direction} {SYMBOL}, volume={volume}, "
                f"SL={sl}, TP={tp}, ticket={result.order}"
            )
            return

        print(f"Order not filled: {result.retcode} — {result.comment}")

        # Do not blindly retry after an uncertain execution.
        if result.retcode not in (
            mt5.TRADE_RETCODE_INVALID_FILL,
            mt5.TRADE_RETCODE_INVALID,
        ):
            return

    print("No order was confirmed. Check the MT5 Experts and Journal tabs.")


# ========== MAIN LOOP ==========

def main():
    connect_mt5()
    starting_equity = load_daily_baseline()
    print("Bot running. Stop it with Ctrl+C on the computer/VPS.")

    try:
        while True:
            now = datetime.now()

            # Avoid trading over the weekend.
            if now.weekday() >= 5:
                print("Weekend: waiting for the market to reopen.")
                time.sleep(60)
                continue

            if daily_loss_limit_hit(starting_equity):
                print("Trading stopped for the rest of this run.")
                break

            try:
                signal = get_signal()
                if signal:
                    submit_trade(signal)
            except Exception as exc:
                print(f"Cycle error: {exc}")

            time.sleep(CHECK_EVERY_SECONDS)

    except KeyboardInterrupt:
        print("Stop requested by user.")

    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()
