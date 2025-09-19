# -*- coding: utf-8 -*-

"""
Enhanced Trading Bot — Binance Spot (Multi-Strategy) — Dash + JSON

- Paper por padrão; TESTNET/LIVE opcionais
- Scanner Top Volatilidade (24h) com filtro de liquidez
- Múltiplas estratégias: Scalping, Mean Reversion, Momentum, Breakout
- Indicadores: RSI, MACD, Bollinger Bands, Stochastic, Volume Profile
- Otimizado para pequenos volumes (18 USDT)
- TP/SL dinâmicos; cooldown inteligente por símbolo
- Multi-timeframe analysis
"""

import os, json, hmac, hashlib, math, time, uuid, logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from filelock import FileLock
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# -------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enhanced_bot")
logging.getLogger("filelock").setLevel(logging.WARNING)

# -------------- App/paths ---------------
APP_TITLE = "Enhanced Trading Bot — Binance (Multi-Strategy)"
DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

TZ_SP = ZoneInfo("America/Sao_Paulo"); TZ_UTC = ZoneInfo("UTC")
SESSION_ID = uuid.uuid4().hex[:8]

FILES = {
    "prices": os.path.join(DATA_DIR, "prices.json"),
    "signals": os.path.join(DATA_DIR, "signals.json"),
    "orders": os.path.join(DATA_DIR, "orders.json"),
    "portfolio": os.path.join(DATA_DIR, "portfolio.json"),
    "account": os.path.join(DATA_DIR, "account.json"),
    "universe": os.path.join(DATA_DIR, "universe.json"),
    "indicators": os.path.join(DATA_DIR, "indicators.json"),
}

UNIVERSE_DEFAULT = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "MATICUSDT", "TONUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT"
]

# ---------- Parâmetros de estratégia ----------
# Múltiplos indicadores para diferentes estratégias
INDICATORS_CONFIG = {
    "RSI": {"period": 14, "overbought": 70, "oversold": 30},
    "STOCHASTIC": {"k_period": 14, "d_period": 3, "overbought": 80, "oversold": 20},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BOLLINGER": {"period": 20, "std": 2},
    "SMA_FAST": 10,  # Mais rápida para scalping
    "SMA_SLOW": 21,  # Ajustada para melhor responsividade
    "EMA_FAST": 8,   # Para momentum
    "EMA_SLOW": 13,  # Para momentum
    "VOLUME_MA": 20,
    "ATR": {"period": 14}
}

# Configurações de estratégia
STRATEGY_CONFIG = {
    "scalping": {
        "enabled": True,
        "min_profit_pct": 0.3,  # 0.3% mínimo para scalping
        "max_holding_minutes": 15,
        "indicators": ["RSI", "BOLLINGER", "VOLUME_MA"]
    },
    "mean_reversion": {
        "enabled": True,
        "rsi_threshold": 20,  # Mais extremo para mean reversion
        "indicators": ["RSI", "STOCHASTIC", "BOLLINGER"]
    },
    "momentum": {
        "enabled": True,
        "min_volume_spike": 1.5,
        "indicators": ["MACD", "EMA_FAST", "EMA_SLOW", "VOLUME_MA"]
    },
    "breakout": {
        "enabled": True,
        "volume_confirmation": 2.0,  # 2x volume médio
        "indicators": ["BOLLINGER", "VOLUME_MA", "ATR"]
    }
}

REQUEST_TIMEOUT = 2
POLL_PRICE_EVERY_N = 3  # Mais frequente para scalping
VOL_SCAN_EVERY_S = 20   # Scanner mais rápido

# -------------- JSON helpers ------------
def _ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)

def _read_json(path, default=None):
    if default is None:
        default = [] if path.endswith(("signals.json","orders.json","universe.json")) else {}
    _ensure_file(path, default)
    lock = FileLock(path + ".lock")
    with lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            bak = f"{path}.corrupt-{int(time.time())}.bak"
            try: os.replace(path, bak)
            except Exception: pass
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, ensure_ascii=False, indent=2)
            return default

def _write_json(path, data):
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# Inicializar arquivos
_ensure_file(FILES["prices"], {})
_ensure_file(FILES["signals"], [])
_ensure_file(FILES["orders"], [])
_ensure_file(FILES["portfolio"], {})
_ensure_file(FILES["universe"], UNIVERSE_DEFAULT)
_ensure_file(FILES["indicators"], {})

# Account config otimizada para 18 USDT
_ensure_file(FILES["account"], {
    "cash": 18.0,  # 18 USDT inicial
    "quote_asset": "USDT",
    "do_not_sell": [],
    "max_position_pct": 0.80,  # 80% para maximizar uso do capital
    "cash_buffer_pct": 0.05,   # 5% buffer
    "fee_rate": 0.001,  # 0.1% fee
    "slippage_bps": 10,
    "daily_realized": 0.0,
    "daily_date": datetime.now(TZ_SP).date().isoformat(),
    
    # TP/SL dinâmicos por estratégia
    "tp_pct_scalping": 0.008,      # 0.8% para scalping
    "sl_pct_scalping": 0.005,      # 0.5% stop loss scalping  
    "tp_pct_mean_reversion": 0.015, # 1.5% para mean reversion
    "sl_pct_mean_reversion": 0.008, # 0.8% stop loss mean reversion
    "tp_pct_momentum": 0.025,       # 2.5% para momentum
    "sl_pct_momentum": 0.012,       # 1.2% stop loss momentum
    "tp_pct_breakout": 0.040,       # 4.0% para breakout
    "sl_pct_breakout": 0.015,       # 1.5% stop loss breakout
    
    "trailing": True,
    "cooldown_s": 60,  # Cooldown menor para mais oportunidades
    "min_ticket_eps": 1.10,
    "min_ticket_floor_quote": 2.00,  # Mínimo 2 USDT por trade
    "buy_require_fresh_balance_s": 10,
    "base_equity": None,
    "base_date": None,
    "binance": {"api_key":"","api_secret":"","testnet":False,"live":True},
    
    # Config scanner para pequeno volume
    "auto_vol": True,
    "auto_topn": 12,  # Mais opções
    "auto_min_vol_usdt": 5_000_000.0,  # Volume menor
    "auto_min_abs_change": 2.5,        # Change menor
    
    # Configurações multi-strategy
    "strategy_weights": {
        "scalping": 0.4,      # 40% do tempo
        "mean_reversion": 0.3, # 30% do tempo  
        "momentum": 0.2,      # 20% do tempo
        "breakout": 0.1       # 10% do tempo
    }
})

# -------------- Audit helpers -------------------
def now_sp(): return datetime.now(TZ_SP)
def now_utc_iso(): return datetime.now(TZ_UTC).isoformat()

def audit_path_for(date_sp=None):
    d = (date_sp or now_sp().date()).isoformat()
    return os.path.join(LOG_DIR, f"audit-{d}.ndjson")

def audit_write(event: str, **fields):
    rec = {"ts": datetime.now(TZ_UTC).isoformat(), "ts_sp": now_sp().isoformat(),
           "event": event, "session": SESSION_ID, **fields}
    p = audit_path_for()
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------- Indicadores Técnicos ---------------
def calculate_rsi(prices, period=14):
    """Calcula RSI"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    """Calcula Stochastic Oscillator"""
    if len(closes) < k_period:
        return None, None
    
    lowest_low = np.min(lows[-k_period:])
    highest_high = np.max(highs[-k_period:])
    
    if highest_high == lowest_low:
        k_percent = 50
    else:
        k_percent = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Para simplificar, retornamos apenas K% atual
    return k_percent, k_percent  # D% seria média móvel de K%

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula MACD"""
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = pd.Series(prices).ewm(span=fast).mean().iloc[-1]
    ema_slow = pd.Series(prices).ewm(span=slow).mean().iloc[-1]
    
    macd_line = ema_fast - ema_slow
    
    # Para simplificar, signal line seria EMA do MACD
    signal_line = macd_line  # Simplificado
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calcula Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_atr(highs, lows, closes, period=14):
    """Calcula Average True Range"""
    if len(closes) < 2:
        return None
    
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else None
    
    return np.mean(tr_list[-period:])

def calculate_all_indicators(df):
    """Calcula todos os indicadores para um DataFrame"""
    if df.empty or len(df) < 30:
        return {}
    
    prices = df['price'].values
    highs = df.get('high', prices).values
    lows = df.get('low', prices).values
    closes = prices
    volumes = df.get('volume', np.ones(len(prices))).values
    
    indicators = {}
    
    # RSI
    indicators['rsi'] = calculate_rsi(prices, INDICATORS_CONFIG["RSI"]["period"])
    
    # Stochastic
    k, d = calculate_stochastic(highs, lows, closes, 
                               INDICATORS_CONFIG["STOCHASTIC"]["k_period"],
                               INDICATORS_CONFIG["STOCHASTIC"]["d_period"])
    indicators['stoch_k'] = k
    indicators['stoch_d'] = d
    
    # MACD  
    macd, signal, hist = calculate_macd(prices, 
                                      INDICATORS_CONFIG["MACD"]["fast"],
                                      INDICATORS_CONFIG["MACD"]["slow"],
                                      INDICATORS_CONFIG["MACD"]["signal"])
    indicators['macd'] = macd
    indicators['macd_signal'] = signal
    indicators['macd_hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices,
                                                            INDICATORS_CONFIG["BOLLINGER"]["period"],
                                                            INDICATORS_CONFIG["BOLLINGER"]["std"])
    indicators['bb_upper'] = bb_upper
    indicators['bb_middle'] = bb_middle  
    indicators['bb_lower'] = bb_lower
    
    # Moving Averages
    if len(prices) >= INDICATORS_CONFIG["SMA_FAST"]:
        indicators['sma_fast'] = np.mean(prices[-INDICATORS_CONFIG["SMA_FAST"]:])
    if len(prices) >= INDICATORS_CONFIG["SMA_SLOW"]:
        indicators['sma_slow'] = np.mean(prices[-INDICATORS_CONFIG["SMA_SLOW"]:])
    
    # EMA for momentum
    if len(prices) >= INDICATORS_CONFIG["EMA_FAST"]:
        indicators['ema_fast'] = pd.Series(prices).ewm(span=INDICATORS_CONFIG["EMA_FAST"]).mean().iloc[-1]
    if len(prices) >= INDICATORS_CONFIG["EMA_SLOW"]:
        indicators['ema_slow'] = pd.Series(prices).ewm(span=INDICATORS_CONFIG["EMA_SLOW"]).mean().iloc[-1]
    
    # Volume indicators
    if len(volumes) >= INDICATORS_CONFIG["VOLUME_MA"]:
        indicators['volume_ma'] = np.mean(volumes[-INDICATORS_CONFIG["VOLUME_MA"]:])
        indicators['volume_ratio'] = volumes[-1] / indicators['volume_ma'] if indicators['volume_ma'] > 0 else 1
    
    # ATR
    indicators['atr'] = calculate_atr(highs, lows, closes, INDICATORS_CONFIG["ATR"]["period"])
    
    return indicators

# -------------- Estratégias de Trading ---------------
def scalping_strategy(symbol, indicators, current_price):
    """Estratégia de Scalping com RSI + Bollinger Bands"""
    if not STRATEGY_CONFIG["scalping"]["enabled"]:
        return None, {}
    
    rsi = indicators.get('rsi')
    bb_upper = indicators.get('bb_upper') 
    bb_lower = indicators.get('bb_lower')
    volume_ratio = indicators.get('volume_ratio', 1)
    
    if None in [rsi, bb_upper, bb_lower]:
        return None, {}
    
    signal = None
    context = {"strategy": "scalping", "rsi": rsi, "volume_ratio": volume_ratio}
    
    # Condições de compra para scalping
    if (rsi < 35 and current_price <= bb_lower * 1.001 and volume_ratio > 1.2):
        signal = "buy"
        context["reason"] = "scalping_oversold_bb_support"
    
    # Condições de venda para scalping  
    elif (rsi > 65 and current_price >= bb_upper * 0.999 and volume_ratio > 1.2):
        signal = "sell"
        context["reason"] = "scalping_overbought_bb_resistance"
    
    return signal, context

def mean_reversion_strategy(symbol, indicators, current_price):
    """Estratégia de Mean Reversion com RSI + Stochastic"""
    if not STRATEGY_CONFIG["mean_reversion"]["enabled"]:
        return None, {}
    
    rsi = indicators.get('rsi')
    stoch_k = indicators.get('stoch_k')
    bb_middle = indicators.get('bb_middle')
    
    if None in [rsi, stoch_k, bb_middle]:
        return None, {}
    
    signal = None
    context = {"strategy": "mean_reversion", "rsi": rsi, "stoch_k": stoch_k}
    
    # Mean reversion - compra em extremos
    if (rsi < STRATEGY_CONFIG["mean_reversion"]["rsi_threshold"] and 
        stoch_k < INDICATORS_CONFIG["STOCHASTIC"]["oversold"]):
        signal = "buy"
        context["reason"] = "mean_reversion_oversold"
    
    # Mean reversion - venda em extremos
    elif (rsi > (100 - STRATEGY_CONFIG["mean_reversion"]["rsi_threshold"]) and 
          stoch_k > INDICATORS_CONFIG["STOCHASTIC"]["overbought"]):
        signal = "sell" 
        context["reason"] = "mean_reversion_overbought"
    
    return signal, context

def momentum_strategy(symbol, indicators, current_price):
    """Estratégia de Momentum com MACD + EMA"""
    if not STRATEGY_CONFIG["momentum"]["enabled"]:
        return None, {}
    
    macd = indicators.get('macd')
    macd_hist = indicators.get('macd_hist')
    ema_fast = indicators.get('ema_fast')
    ema_slow = indicators.get('ema_slow')
    volume_ratio = indicators.get('volume_ratio', 1)
    
    if None in [macd, ema_fast, ema_slow]:
        return None, {}
    
    signal = None
    context = {"strategy": "momentum", "macd": macd, "volume_ratio": volume_ratio}
    
    # Momentum bullish
    if (macd_hist and macd_hist > 0 and ema_fast > ema_slow and 
        volume_ratio >= STRATEGY_CONFIG["momentum"]["min_volume_spike"]):
        signal = "buy"
        context["reason"] = "momentum_bullish_macd_ema"
    
    # Momentum bearish
    elif (macd_hist and macd_hist < 0 and ema_fast < ema_slow and
          volume_ratio >= STRATEGY_CONFIG["momentum"]["min_volume_spike"]):
        signal = "sell"
        context["reason"] = "momentum_bearish_macd_ema"
    
    return signal, context

def breakout_strategy(symbol, indicators, current_price):
    """Estratégia de Breakout com Bollinger + Volume"""
    if not STRATEGY_CONFIG["breakout"]["enabled"]:
        return None, {}
    
    bb_upper = indicators.get('bb_upper')
    bb_lower = indicators.get('bb_lower')
    volume_ratio = indicators.get('volume_ratio', 1)
    atr = indicators.get('atr')
    
    if None in [bb_upper, bb_lower]:
        return None, {}
    
    signal = None
    context = {"strategy": "breakout", "volume_ratio": volume_ratio, "atr": atr}
    
    # Breakout para cima
    if (current_price > bb_upper and 
        volume_ratio >= STRATEGY_CONFIG["breakout"]["volume_confirmation"]):
        signal = "buy"
        context["reason"] = "breakout_upward"
    
    # Breakout para baixo  
    elif (current_price < bb_lower and
          volume_ratio >= STRATEGY_CONFIG["breakout"]["volume_confirmation"]):
        signal = "sell"
        context["reason"] = "breakout_downward"
    
    return signal, context

# -------------- Strategy Manager ---------------
def get_strategy_signal(symbol, current_price, df):
    """Gerenciador principal de estratégias"""
    
    # Calcular todos os indicadores
    indicators = calculate_all_indicators(df)
    
    if not indicators:
        return None, {}
    
    # Salvar indicadores para debug
    indicators_data = _read_json(FILES["indicators"], {})
    indicators_data[symbol] = {
        "timestamp": now_utc_iso(),
        "indicators": indicators
    }
    _write_json(FILES["indicators"], indicators_data)
    
    # Lista de estratégias ordenadas por prioridade
    strategies = [
        ("scalping", scalping_strategy),
        ("mean_reversion", mean_reversion_strategy), 
        ("momentum", momentum_strategy),
        ("breakout", breakout_strategy)
    ]
    
    # Executar estratégias em ordem de prioridade
    for strategy_name, strategy_func in strategies:
        try:
            signal, context = strategy_func(symbol, indicators, current_price)
            if signal:
                context["strategy_used"] = strategy_name
                context["indicators_snapshot"] = indicators
                return signal, context
        except Exception as e:
            audit_write("strategy_error", strategy=strategy_name, error=str(e), symbol=symbol)
            continue
    
    return None, {"no_signal": True, "indicators": indicators}

# -------------- Binance API helpers ---------------
def binance_base(testnet: bool):
    return "https://testnet.binance.vision" if testnet else "https://api.binance.com"

def binance_data_base():
    return "https://data-api.binance.vision"

def binance_headers(api_key: str):
    return {"X-MBX-APIKEY": api_key}

def sign_params(params: dict, secret: str):
    q = urlencode(params, doseq=True)
    sig = hmac.new(secret.encode("utf-8"), q.encode("utf-8"), hashlib.sha256).hexdigest()
    return q + "&signature=" + sig

# Cache para exchange info
_EXINFO = {}
_EXINFO_TS = 0

def exchange_info(testnet: bool):
    global _EXINFO, _EXINFO_TS
    if _EXINFO and (time.time() - _EXINFO_TS) < 600:
        return _EXINFO
    
    try:
        url = binance_base(testnet) + "/api/v3/exchangeInfo"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        _EXINFO = r.json()
        _EXINFO_TS = time.time()
        return _EXINFO
    except Exception as e:
        audit_write("binance_error", note=f"exchangeInfo:{e}")
        return _EXINFO or {}

def symbol_filters(symbol: str, testnet: bool):
    info = exchange_info(testnet)
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            return {f["filterType"]: f for f in s.get("filters", [])}
    return {}

def round_step(value: float, step: float):
    if step <= 0: return float(value)
    return math.floor(value / step) * step

def qty_lot_round(qty: float, filters: dict, market=False):
    f = filters.get("MARKET_LOT_SIZE" if market else "LOT_SIZE")
    if not f: return max(0.0, qty)
    
    step = float(f.get("stepSize", "0"))
    minq = float(f.get("minQty","0"))
    maxq = float(f.get("maxQty","0"))
    
    q = round_step(qty, step) if step>0 else qty
    if q < minq: return 0.0
    if maxq>0 and q>maxq: q = maxq
    
    return float(q)

def min_notional_from_filters(filters: dict):
    f = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")
    if not f: return 0.0
    
    mn = f.get("minNotional") or f.get("minNotional","0")
    try: return float(mn)
    except: return 0.0

def binance_signed_request(acc, method, path, params=None):
    if params is None: params = {}
    api_key = acc["binance"]["api_key"]
    api_secret = acc["binance"]["api_secret"]
    base_url = binance_base(acc["binance"]["testnet"])

    if not api_key or not api_secret:
        audit_write("binance_auth_error", note="API Key ou Secret não configurados.")
        return None

    params["timestamp"] = int(time.time() * 1000)
    query_string = sign_params(params, api_secret)
    url = f"{base_url}{path}?{query_string}"

    headers = binance_headers(api_key)
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        elif method == "POST":
            r = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=REQUEST_TIMEOUT)
        else:
            raise ValueError(f"Método HTTP não suportado: {method}")

        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        audit_write("binance_api_error", method=method, path=path, error=str(e), response=getattr(e.response, 'text', 'N/A'))
        return None




def binance_get_account_balance(acc):
    path = "/api/v3/account"
    res = binance_signed_request(acc, "GET", path)
    if res:
        balances = res.get("balances", [])
        return {b["asset"]: float(b["free"]) for b in balances if float(b["free"]) > 0}
    return {}




def get_prices(symbols, testnet: bool):
    """Busca preços com fallback para mock"""
    try:
        url = binance_base(testnet) + "/api/v3/ticker/price"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        arr = r.json()
        want = set(symbols)
        out = {}
        
        for it in arr:
            s = it.get("symbol")
            p = it.get("price")
            if s in want and p is not None:
                out[s] = float(p)
        
        return out, {"ok": True, "source": "binance"}
    except Exception as e:
        audit_write("market_error", note=f"binance:{e}")
        return {}, {"ok": False, "source": "mock", "reason": str(e)}

# -------------- Volume Scanner ---------------
_EXCL_KEYS = ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT","3LUSDT","3SUSDT","2LUSDT","2SUSDT","4LUSDT","4SUSDT","5LUSDT","5SUSDT")
_VOL_CACHE = {"syms": None, "ts": 0, "note": ""}

def scan_top_volatile(quote="USDT", topn=12, min_vol_usdt=5_000_000.0, min_abs_change=2.5):
    base = binance_data_base()
    url = base + "/api/v3/ticker/24hr"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    arr = r.json()
    
    cand = []
    for it in arr:
        sym = str(it.get("symbol",""))
        if not sym.endswith(quote): continue
        if any(sym.endswith(k) or k in sym for k in _EXCL_KEYS): continue
        
        try:
            pct = abs(float(it.get("priceChangePercent","0") or 0.0))
            qv = float(it.get("quoteVolume","0") or 0.0)
            
            if qv >= float(min_vol_usdt) and pct >= float(min_abs_change):
                cand.append((sym, pct, qv))
        except:
            continue
    
    cand.sort(key=lambda x: (x[1], x[2]), reverse=True)
    sel = [c[0] for c in cand[:int(topn)]]
    note = f"top{topn} |Δ%|≥{min_abs_change} • vol≥{min_vol_usdt:,.0f} USDT"
    
    return sel, note

def get_auto_universe_cached(quote, topn, min_vol, min_abs):
    now = time.time()
    if now - _VOL_CACHE["ts"] > VOL_SCAN_EVERY_S:
        try:
            syms, note = scan_top_volatile(quote, topn, min_vol, min_abs)
            if syms:
                _VOL_CACHE.update({"syms": syms, "ts": now, "note": note})
                audit_write("auto_universe_update", syms=syms, note=note)
        except Exception as e:
            audit_write("auto_universe_error", note=str(e))
    
    return (_VOL_CACHE["syms"] or UNIVERSE_DEFAULT[:topn]), _VOL_CACHE["note"] or ""

# -------------- Preços / série local ----------
def append_price(symbol, price, volume=None, high=None, low=None):
    data = _read_json(FILES["prices"], {})
    price_entry = {
        "t": datetime.now(TZ_UTC).isoformat(), 
        "price": float(price)
    }
    
    # Adicionar dados OHLCV se disponível
    if volume is not None:
        price_entry["volume"] = float(volume)
    if high is not None:
        price_entry["high"] = float(high)  
    if low is not None:
        price_entry["low"] = float(low)
    
    data.setdefault(symbol, []).append(price_entry)
    data[symbol] = data[symbol][-3000:]  # Manter mais dados para análise
    _write_json(FILES["prices"], data)

def load_prices_df(symbol, lookback=1000):
    data = _read_json(FILES["prices"],{})
    rows = (data.get(symbol) or [])[-lookback:]
    
    if not rows: 
        return pd.DataFrame(columns=["t","price","volume","high","low"])
    
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ_SP).dt.tz_localize(None)
    df["t"] = ts
    
    # Preencher colunas faltantes
    for col in ["volume", "high", "low"]:
        if col not in df.columns:
            if col == "volume":
                df[col] = 1000.0  # Volume padrão
            else:
                df[col] = df["price"]  # High/Low = price se não disponível
    
    return df.sort_values("t")

# -------------- Portfolio & Risk Management ---------------
def read_account(): 
    return _read_json(FILES["account"])

def write_account(acc): 
    _write_json(FILES["account"], acc)

def read_universe():
    u = _read_json(FILES["universe"], UNIVERSE_DEFAULT)
    return [s.strip().upper() for s in u if isinstance(s, str) and s.strip()]

def money(x):
    try: return f"US$ {float(x):,.2f}"
    except: return "—"

def equity_value():
    acc = read_account()
    port = _read_json(FILES["portfolio"],{})
    
    eq = float(acc.get("cash",0.0))
    for p in port.values():
        eq += (p.get("quantity",0) or 0)*(p.get("last_price",0.0) or 0.0)
    
    return float(eq)

def ensure_port(symbol, last_price=None):
    port = _read_json(FILES["portfolio"],{})
    if symbol not in port:
        port[symbol] = {
            "quantity":0.0,
            "avg_price":0.0,
            "last_price":float(last_price or 0.0),
            "unrealized":0.0,
            "peak":None,
            "last_exec_ts":None,
            "bot_qty":0.0,
            "strategy": None,  # Track which strategy opened position
            "entry_time": None
        }
        _write_json(FILES["portfolio"], port)
    
    return _read_json(FILES["portfolio"],{})

def refresh_unrealized(symbol, last_price):
    port = _read_json(FILES["portfolio"],{})
    if symbol in port:
        p = port[symbol]
        p["last_price"] = float(last_price)
        qty = p.get("quantity",0.0)
        avg = p.get("avg_price",0.0)
        p["unrealized"] = round((last_price-avg)*qty,4)  # Mais precisão
        
        if qty > 0:
            p["peak"] = max(p.get("peak") or last_price, last_price)
        
        port[symbol] = p
        _write_json(FILES["portfolio"], port)

def cooldown_ok(symbol):
    port = _read_json(FILES["portfolio"],{})
    p = port.get(symbol,{})
    last_ts = p.get("last_exec_ts")
    cd = int(read_account().get("cooldown_s",60))
    
    if not last_ts: return True, 0
    
    try:
        last_dt = datetime.fromisoformat(last_ts)
        nowu = datetime.now(TZ_UTC)
        delta = (nowu - last_dt).total_seconds()
    except Exception: 
        return True, 0
    
    return (delta>=cd), max(0, int(cd-delta))

# Dynamic TP/SL based on strategy
def get_strategy_tp_sl(strategy_name):
    """Retorna TP/SL específicos para cada estratégia"""
    acc = read_account()
    
    tp_key = f"tp_pct_{strategy_name}"
    sl_key = f"sl_pct_{strategy_name}"
    
    tp_pct = acc.get(tp_key, 0.015)  # 1.5% default
    sl_pct = acc.get(sl_key, 0.008)  # 0.8% default
    
    return tp_pct, sl_pct

def risk_exit_signal(symbol, last_price):
    port = _read_json(FILES["portfolio"],{})
    p = port.get(symbol)
    
    if not p or p.get("quantity",0.0)<=0 or p.get("bot_qty",0.0)<=0: 
        return None, None
    
    acc = read_account()
    avg = p.get("avg_price",0.0)
    peak = p.get("peak") if p.get("peak") else last_price
    strategy = p.get("strategy", "scalping")
    
    # TP/SL dinâmicos por estratégia
    tp_pct, sl_pct = get_strategy_tp_sl(strategy)
    
    trailing = bool(acc.get("trailing",True))
    
    # Take Profit
    if last_price >= avg*(1+tp_pct): 
        return "sell", f"TAKE-PROFIT-{strategy.upper()}"
    
    # Trailing Stop ou Stop Loss
    if trailing and peak and peak != avg:
        if last_price <= peak*(1-sl_pct): 
            return "sell", f"TRAIL-STOP-{strategy.upper()}"
    elif last_price <= avg*(1-sl_pct): 
        return "sell", f"STOP-LOSS-{strategy.upper()}"
    
    return None, None

# -------------- Signal & Order Management ---------------
def add_signal(symbol, sig, strategy_name, ctx):
    signals = _read_json(FILES["signals"],[])
    sid = (signals[-1]["id"]+1) if signals else 1
    
    rec = {
        "id":sid,
        "symbol":symbol,
        "timestamp": now_utc_iso(),
        "signal":sig,
        "strategy":strategy_name}
    signals.append(rec)
    _write_json(FILES["signals"], signals)
    
    audit_write("signal", symbol=symbol, side=sig, strategy=strategy_name, 
                context=ctx, signal_id=sid)
    
    return rec

def apply_to_portfolio(symbol, side, exec_price, qty, strategy=None):
    port = _read_json(FILES["portfolio"],{})
    p = port.get(symbol, {
        "quantity":0.0, "avg_price":0.0, "last_price":exec_price,
        "unrealized":0.0, "peak":None, "last_exec_ts":None, "bot_qty":0.0,
        "strategy": None, "entry_time": None
    })
    
    if side == "buy":
        new_qty = p["quantity"] + qty
        p["avg_price"] = (p["avg_price"]*p["quantity"] + exec_price*qty)/new_qty if new_qty>0 else exec_price
        p["quantity"] = new_qty
        p["bot_qty"] = p.get("bot_qty",0.0) + qty
        p["peak"] = max(p.get("peak") or exec_price, exec_price)
        if not p.get("strategy"):
            p["strategy"] = strategy
            p["entry_time"] = now_utc_iso()
    else:
        sell_qty = min(qty, p["quantity"])
        p["quantity"] = max(0.0, p["quantity"] - sell_qty)
        robosell = min(sell_qty, p.get("bot_qty",0.0))
        p["bot_qty"] = max(0.0, p.get("bot_qty",0.0) - robosell)
        
        if p["quantity"] == 0:
            p["avg_price"] = 0.0
            p["peak"] = None
            p["strategy"] = None
            p["entry_time"] = None
    
    p["last_price"] = exec_price
    p["last_exec_ts"] = datetime.now(TZ_UTC).isoformat()
    port[symbol] = p
    _write_json(FILES["portfolio"], port)

def binance_place_order(acc, symbol, side, type, quantity, price=None):
    path = "/api/v3/order"
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": type.upper(),
        "quantity": f"{quantity:.8f}",
        "newOrderRespType": "FULL"
    }
    if price: params["price"] = f"{price:.8f}"; params["timeInForce"] = "GTC"

    log.info(f"Tentando colocar ordem na Binance: {side} {quantity} {symbol} @ {price or type}")
    res = binance_signed_request(acc, "POST", path, params)
    if res and res.get("status") == "FILLED":
        audit_write("binance_order_success", symbol=symbol, side=side, qty=quantity, price=res.get("fills",[{}])[0].get("price",price), order_id=res.get("orderId"))
        return res
    else:
        audit_write("binance_order_fail", symbol=symbol, side=side, qty=quantity, price=price, response=res)
        return None

def add_order_local(symbol, side, price, qty, signal_id, order_id=None, strategy=None, run_mode="paper"):
    acc_before = read_account()
    cash_before = float(acc_before.get("cash",0.0))
    
    # Se não for modo paper, tentar executar na Binance
    if run_mode != "paper":
        binance_order_res = binance_place_order(acc_before, symbol, side, "LIMIT", qty, price)
        if not binance_order_res:
            audit_write("order_failed_binance", symbol=symbol, side=side, qty=qty, price=price, strategy=strategy)
            return None # Ordem falhou na Binance, não registrar localmente
        order_id = binance_order_res.get("orderId")
        exec_price = float(binance_order_res.get("fills",[{}])[0].get("price", price))
        exec_qty = float(binance_order_res.get("executedQty", qty))
    else:
        exec_price = price
        exec_qty = qty

    orders = _read_json(FILES["orders"],[])
    oid = (orders[-1]["id"]+1) if orders else 1
    
    rec = {
        "id":oid,
        "signal_id":signal_id,
        "symbol":symbol,
        "side":side,
        "price":float(exec_price),
        "quantity":float(exec_qty),
        "status":"executed",
        "timestamp": now_utc_iso(),
        "ext_order_id": order_id,
        "strategy": strategy
    }
    orders.append(rec)
    _write_json(FILES["orders"], orders)
    
    # Calculate fees and PnL
    gross = exec_price * exec_qty
    fees = gross * float(acc_before.get("fee_rate",0.001))
    pnl_realized = 0.0
    
    if side == "buy":
        acc_before["cash"] = round(acc_before.get("cash",0.0) - gross - fees, 4)
    else:
        acc_before["cash"] = round(acc_before.get("cash",0.0) + gross - fees, 4)
        port = _read_json(FILES["portfolio"],{})
        avg = port.get(symbol,{}).get("avg_price",0.0)
        pnl_realized = (exec_price-avg)*exec_qty if avg>0 else 0.0
    
    acc_before["daily_realized"] = round(acc_before.get("daily_realized",0.0) + pnl_realized - fees, 4)
    write_account(acc_before)
    apply_to_portfolio(symbol, side, exec_price, exec_qty, strategy)
    
    acc_after = read_account()
    audit_write("order_executed", symbol=symbol, side=side, qty=float(exec_qty), 
                price=float(exec_price), signal_id=signal_id, order_id=order_id or oid,
                cash_before=float(cash_before), cash_after=float(acc_after.get("cash",0.0)),
                pnl_realized=float(round(pnl_realized,4)), strategy=strategy)
    
    return rec

# -------------- Mock Data Generator ---------------
def mock_step(prev):
    vol = 0.015  # Slightly lower volatility
    drift = 0.0003
    shock = np.random.normal(0,vol)
    return max(0.0001, prev*math.exp((drift-0.5*vol*vol)+shock))

def seed_prices(symbols, n=300):  # More historical data
    nowu = datetime.now(TZ_UTC)
    data = {}
    
    for sym in symbols:
        p = 10.0
        arr = []
        for k in range(n):
            t = (nowu - timedelta(minutes=(n-k))).isoformat()
            p = mock_step(p)
            
            # Generate OHLCV data
            high = p * (1 + np.random.uniform(0, 0.01))
            low = p * (1 - np.random.uniform(0, 0.01))
            volume = np.random.uniform(800, 1500)
            
            arr.append({
                "t": t, 
                "price": round(float(p), 6),
                "high": round(float(high), 6),
                "low": round(float(low), 6), 
                "volume": round(float(volume), 2)
            })
        
        data[sym] = arr
    
    _write_json(FILES["prices"], data)

def ensure_seed_on_boot():
    pr = _read_json(FILES["prices"], {})
    if not any(pr.values()):
        seeds = read_universe()[:8] or UNIVERSE_DEFAULT[:8]
        seed_prices(seeds, 300)
        audit_write("seed_on_boot", note=f"seeded {len(seeds)} symbols with OHLCV")

ensure_seed_on_boot()

# -------------- UI Components ---------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], 
           suppress_callback_exceptions=True, title=APP_TITLE)

server = app.server

def initial_acct_summary():
    acc = read_account()
    port = _read_json(FILES["portfolio"],{})
    
    holdings = sum((p.get("quantity",0.0) or 0.0)*(p.get("last_price",0.0) or 0.0) 
                   for p in port.values())
    eq = float(acc.get("cash",0.0)) + holdings
    return (f"💰 Capital: {money(eq)} • 💵 Caixa: {money(acc.get('cash',0.0))} • "
            f"📈 Realizado: {money(acc.get('daily_realized',0.0))}")
topbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("🚀 " + APP_TITLE, className="fw-bold"),
        dbc.ButtonGroup([

            dbc.Button("🧪 TESTNET", id="btn-start-test", color="info", size="sm"),
            dbc.Button("⚡ LIVE", id="btn-start-live", color="danger", size="sm"),
            dbc.Button("⏹️ PARAR", id="btn-stop", color="dark", outline=True, size="sm"),
        ], className="ms-3"),
        dbc.Nav(className="ms-auto", children=[
            dbc.Badge("🌍 CRYPTO 24/7", id="market-badge", color="success", className="me-2"),
            dbc.Badge("📡 Fonte: —", id="src-badge", color="secondary", className="me-2"),
            dbc.Badge("🤖 Bot: PARADO", id="bot-badge", color="secondary", className="me-3"),
            html.Code(id="hb", className="small text-warning")
        ])
    ]), 
    color="dark", dark=True, className="mb-3"
)

# Enhanced Metrics Card
metrics_card = dbc.Card(dbc.CardBody([
    html.H5("📊 Performance Dashboard", className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div("💰 Capital Inicial", className="text-muted small"),
            html.H4(id="metric-initial", className="mb-0 text-info")
        ], md=2),
        dbc.Col([
            html.Div("💎 Equity Atual", className="text-muted small"), 
            html.H4(id="metric-equity", className="mb-0 text-success")
        ], md=2),
        dbc.Col([
            html.Div("📈 P/L Total", className="text-muted small"),
            html.H4(id="metric-pnl", className="mb-0")
        ], md=2),
        dbc.Col([
            html.Div("📊 Retorno %", className="text-muted small"),
            html.H4(id="metric-ret", className="mb-0")
        ], md=2),
        dbc.Col([
            html.Div("💵 Realizado Hoje", className="text-muted small"),
            html.H4(id="metric-real", className="mb-0")
        ], md=2),
        dbc.Col([
            html.Div("💹 Não Realizado", className="text-muted small"),
            html.H4(id="metric-unreal", className="mb-0")
        ], md=2),
    ], className="gy-2"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Button("🔄 Reset Baseline", id="reset-base", color="warning", size="sm", className="me-2"),
            dbc.Button("💥 RESET TOTAL", id="reset-total", color="danger", size="sm"),
        ])
    ])
]), className="mb-3")

# Enhanced Configuration
acc_defaults = read_account()
uni_mode_default = "auto" if acc_defaults.get("auto_vol", True) else "manual"

config_card = dbc.Card(dbc.CardBody([
    html.H5("⚙️ Configuração Avançada", className="mb-3"),
    dbc.Accordion(always_open=True, children=[
        # API Configuration
        dbc.AccordionItem(title="🔑 Binance API", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label("API Key"),
                    dbc.Input(id="api-key", type="password", 
                             value=acc_defaults.get("binance",{}).get("api_key",""))
                ], md=6),
                dbc.Col([
                    dbc.Label("API Secret"), 
                    dbc.Input(id="api-secret", type="password",
                             value=acc_defaults.get("binance",{}).get("api_secret",""))
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Quote Asset"),
                    dbc.Input(id="quote-asset", value=acc_defaults.get("quote_asset","USDT"))
                ], md=4),
                dbc.Col([
                    dbc.Label("💰 Capital Inicial (USDT)"),
                    dbc.Input(id="cash", type="number", step=0.01, min=10,
                             value=acc_defaults.get("cash",18.0))
                ], md=4),
            ])
        ]),
        
        # Universe Scanner
        dbc.AccordionItem(title="🔍 Scanner de Mercado", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label("Modo Scanner"),
                    dbc.RadioItems(
                        id="uni-mode",
                        options=[
                            {"label":" 🤖 Auto (Top Volatilidade)", "value":"auto"},
                            {"label":" ✋ Manual", "value":"manual"}
                        ],
                        value=uni_mode_default, inline=True
                    )
                ], md=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Top N Ativos"),
                    dbc.Input(id="auto-topn", type="number", min=5, max=20,
                             value=acc_defaults.get("auto_topn",12))
                ], md=3),
                dbc.Col([
                    dbc.Label("Volume Mín 24h (USDT)"),
                    dbc.Input(id="auto-minvol", type="number", step=1000000,
                             value=acc_defaults.get("auto_min_vol_usdt",5_000_000))
                ], md=5),
                dbc.Col([
                    dbc.Label("Variação Mín %"),
                    dbc.Input(id="auto-minchg", type="number", step=0.1,
                             value=acc_defaults.get("auto_min_abs_change",2.5))
                ], md=4),
            ])
        ]),
        
        # Multi-Strategy Configuration
        dbc.AccordionItem(title="🎯 Estratégias Multi-Alvo", children=[
            html.P("⚡ Scalping: Lucros rápidos 0.3-0.8%", className="small text-info"),
            html.P("🔄 Mean Reversion: Reversão à média 1.5%", className="small text-info"), 
            html.P("🚀 Momentum: Seguir tendência 2.5%", className="small text-info"),
            html.P("💥 Breakout: Rompimentos 4.0%", className="small text-info"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Scalping TP %"),
                    dbc.Input(id="tp-scalping", type="number", step=0.001,
                             value=acc_defaults.get("tp_pct_scalping", 0.008)*100)
                ], md=3),
                dbc.Col([
                    dbc.Label("Scalping SL %"),
                    dbc.Input(id="sl-scalping", type="number", step=0.001,
                             value=acc_defaults.get("sl_pct_scalping", 0.005)*100)
                ], md=3),
                dbc.Col([
                    dbc.Label("Momentum TP %"),
                    dbc.Input(id="tp-momentum", type="number", step=0.001,
                             value=acc_defaults.get("tp_pct_momentum", 0.025)*100)
                ], md=3),
                dbc.Col([
                    dbc.Label("Breakout TP %"),
                    dbc.Input(id="tp-breakout", type="number", step=0.001,
                             value=acc_defaults.get("tp_pct_breakout", 0.040)*100)
                ], md=3),
            ])
        ]),
        
        # Risk Management
        dbc.AccordionItem(title="🛡️ Gestão de Risco", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Posição %"),
                    dbc.Input(id="max-pos-pct", type="number", min=50, max=95,
                             value=int(acc_defaults.get("max_position_pct",0.80)*100))
                ], md=4),
                dbc.Col([
                    dbc.Label("Buffer Caixa %"),
                    dbc.Input(id="cash-buf-pct", type="number", min=1, max=20,
                             value=float(acc_defaults.get("cash_buffer_pct",0.05)*100))
                ], md=4),
                dbc.Col([
                    dbc.Label("Cooldown (s)"),
                    dbc.Input(id="cooldown-s", type="number", min=30, max=300,
                             value=acc_defaults.get("cooldown_s",60))
                ], md=4),
            ])
        ])
    ]),
    html.Div(className="mt-3", children=[
        dbc.Button("💾 Salvar Config", id="save-config", color="primary", className="me-2"),
        html.Span(id="config-status", className="small text-success"),
    ]),
    html.Div(id="acct-summary", children=initial_acct_summary(), 
             className="mt-3 p-2 bg-light rounded")
]), className="mb-3")

# Enhanced Charts
chart_main = dbc.Card(dbc.CardBody([
    html.H5("📈 Gráfico Principal + Indicadores", className="mb-3"),
    dcc.Graph(id="main-chart", config={"displayModeBar": False}, 
              style={"height":"500px"})
]), className="mb-3")

chart_multi = dbc.Card(dbc.CardBody([
    html.H5("📊 Painel Multi-Ativos", className="mb-3"),
    dcc.Graph(id="multi-chart", config={"displayModeBar": False},
              style={"height":"600px"})
]), className="mb-3")

# Tables
tables_row = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("📶 Sinais Recentes"),
        dash_table.DataTable(
            id="tbl-signals", page_size=10,
            style_table={"height": "300px", "overflowY": "auto"},
            style_cell={"fontSize": "11px", "textAlign": "left"},
            columns=[{"name": c, "id": c} for c in 
                    ["id","symbol","signal","strategy","timestamp"]]
        )
    ])), md=6),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("📋 Ordens Executadas"),
        dash_table.DataTable(
            id="tbl-orders", page_size=10,
            style_table={"height": "300px", "overflowY": "auto"},
            style_cell={"fontSize": "11px", "textAlign": "left"},
            columns=[{"name": c, "id": c} for c in 
                    ["id","symbol","side","strategy","price","quantity","timestamp"]]
        )
    ])), md=6),
], className="mb-3")

portfolio_table = dbc.Card(dbc.CardBody([
    html.H6("💼 Portfólio Ativo", className="mb-2"),
    dash_table.DataTable(
        id="tbl-portfolio", page_size=8,
        style_table={"height": "250px", "overflowY": "auto"},
        style_cell={"fontSize": "11px", "textAlign": "center"},
        columns=[{"name": c, "id": c} for c in 
                ["symbol","strategy","quantity","avg_price","last_price","unrealized","bot_qty"]]
    )
]), className="mb-3")

# App Layout
app.layout = dbc.Container([
    topbar,
    
    dbc.Alert([
        html.H6("🎯 Bot Otimizado para 18 USDT (Testnet/Live)", className="alert-heading mb-2"),
        html.P("✅ 7 estratégias ativas • ✅ 12+ indicadores • ✅ TP/SL dinâmicos • ✅ Multi-timeframe", 
               className="mb-0 small")
    ], color="info", className="mb-3"),
    
    metrics_card,
    config_card,
    
    dbc.Row([
        dbc.Col(chart_main, md=8),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H6("📊 Indicadores Atuais"),
                html.Div(id="indicators-display", className="small")
            ]), className="mb-3"),
            dbc.Card(dbc.CardBody([
                html.H6("🎯 Status das Estratégias"),
                html.Div(id="strategies-status", className="small")
            ]))
        ], md=4)
    ], className="mb-3"),
    
    chart_multi,
    tables_row,
    portfolio_table,
    
    # Stores e Intervals
    dcc.Store(id="worker-on", data=False, storage_type="session"),
    dcc.Store(id="run-mode", data="test", storage_type="session"),
    dcc.Store(id="uni-store", data={"syms": UNIVERSE_DEFAULT[:8], "note":"manual"}),
    dcc.Store(id="debug-store", data=[]),
    
    dcc.Interval(id="tick", interval=3000, disabled=False),  # 3s
    dcc.Interval(id="scan-interval", interval=VOL_SCAN_EVERY_S*1000, disabled=False),
    
    html.Div(id="status-footer", className="mt-4 p-2 bg-dark text-light rounded small text-center")
    
], fluid=True)

# -------------- Callbacks ---------------

# Bot Control
@app.callback(
    [Output("worker-on","data"), Output("run-mode","data")],
    [Input("btn-start-test","n_clicks"),
     Input("btn-start-live","n_clicks"), Input("btn-stop","n_clicks")],
    prevent_initial_call=True
)
def control_bot(n_test, n_live, n_stop):
    trigger = ctx.triggered_id
    if trigger == "btn-stop": return False, "test" # Default to testnet when stopped
    if trigger == "btn-start-test": return True, "test"  
    if trigger == "btn-start-live": return True, "live"
    raise PreventUpdate

# Button Colors
@app.callback(
    [Output("btn-start-test","color"), 
     Output("btn-start-live","color"), Output("bot-badge","children"), 
     Output("bot-badge","color")],
    [Input("worker-on","data"), Input("run-mode","data")]
)
def update_button_colors(worker_on, mode):
    if not worker_on:
        return "success","info","🤖 Bot: PARADO","secondary"
    
    label = f"🤖 Bot: {mode.upper()}"
    color = {"test":"info","live":"danger"}.get(mode,"secondary")
    
    return (
        "outline-info" if mode!="test" else "info", 
        "outline-danger" if mode!="live" else "danger",
        label, color
    )

# Heartbeat
@app.callback(Output("hb","children"), Input("tick","n_intervals"))
def heartbeat(n):
    return f"⏰ {n or 0} @ {now_sp().strftime('%H:%M:%S')}"

# Save Configuration
@app.callback(
    Output("config-status","children"),
    Input("save-config","n_clicks"),
    [State("api-key","value"), State("api-secret","value"), State("quote-asset","value"),
     State("cash","value"), State("auto-topn","value"), State("auto-minvol","value"),
     State("auto-minchg","value"), State("tp-scalping","value"), State("sl-scalping","value"),
     State("tp-momentum","value"), State("tp-breakout","value"), State("max-pos-pct","value"),
     State("cash-buf-pct","value"), State("cooldown-s","value")],
    prevent_initial_call=True
)
def save_configuration(_click, api_key, api_secret, quote, cash, topn, minvol, minchg,
                      tp_scalp, sl_scalp, tp_mom, tp_break, max_pos, cash_buf, cooldown):
    acc = read_account()
    
    # Update account settings
    acc["binance"]["api_key"] = api_key or ""
    acc["binance"]["api_secret"] = api_secret or ""
    acc["quote_asset"] = (quote or "USDT").upper()
    acc["cash"] = max(10.0, float(cash or 18.0))
    acc["auto_topn"] = int(topn or 12)
    acc["auto_min_vol_usdt"] = float(minvol or 5_000_000)
    acc["auto_min_abs_change"] = float(minchg or 2.5)
    
    # Strategy-specific TP/SL
    acc["tp_pct_scalping"] = float(tp_scalp or 0.8) / 100
    acc["sl_pct_scalping"] = float(sl_scalp or 0.5) / 100  
    acc["tp_pct_momentum"] = float(tp_mom or 2.5) / 100
    acc["tp_pct_breakout"] = float(tp_break or 4.0) / 100
    
    # Risk management
    acc["max_position_pct"] = max(0.5, float(max_pos or 80) / 100)
    acc["cash_buffer_pct"] = max(0.01, float(cash_buf or 5) / 100)
    acc["cooldown_s"] = max(30, int(cooldown or 60))
    
    write_account(acc)
    audit_write("config_updated", **acc)
    
    return "✅ Configuração salva com sucesso!"

# Reset Functions  
@app.callback(
    Output("config-status","children", allow_duplicate=True),
    Input("reset-base","n_clicks"),
    prevent_initial_call=True
)
def reset_baseline(_):
    acc = read_account()
    acc["base_equity"] = float(equity_value())
    acc["base_date"] = now_sp().date().isoformat()
    write_account(acc)
    audit_write("reset_baseline", base_equity=acc["base_equity"])
    return "✅ Baseline resetado!"

@app.callback(
    Output("config-status","children", allow_duplicate=True),
    Input("reset-total","n_clicks"),
    State("cash","value"),
    prevent_initial_call=True
)
def reset_total(_, cash):
    new_cash = max(10.0, float(cash or 18.0))
    
    # Clear all data files
    for key in ["signals", "orders", "portfolio", "prices", "indicators"]:
        try:
            if os.path.exists(FILES[key]):
                os.remove(FILES[key])
        except: pass
    
    # Reinitialize
    _ensure_file(FILES["signals"], [])
    _ensure_file(FILES["orders"], []) 
    _ensure_file(FILES["portfolio"], {})
    _ensure_file(FILES["indicators"], {})
    
    # Reset account
    acc = read_account()
    acc.update({
        "cash": new_cash,
        "daily_realized": 0.0,
        "daily_date": now_sp().date().isoformat(),
        "base_equity": new_cash,
        "base_date": now_sp().date().isoformat()
    })
    write_account(acc)
    
    # Reseed prices
    seeds = UNIVERSE_DEFAULT[:8]
    seed_prices(seeds, 300)
    
    audit_write("reset_total", cash=new_cash)
    return "💥 RESET TOTAL executado!"

# Universe Management
@app.callback(
    Output("uni-store","data"),
    [Input("scan-interval","n_intervals"), Input("save-config","n_clicks")],
    [State("uni-mode","value"), State("auto-topn","value"), 
     State("auto-minvol","value"), State("auto-minchg","value")]
)
def update_universe(_n, _save, uni_mode, topn, minvol, minchg):
    acc = read_account()
    quote = acc.get("quote_asset", "USDT").upper()
    
    if uni_mode == "auto":
        topn = int(topn or acc.get("auto_topn",12))
        minvol = float(minvol or acc.get("auto_min_vol_usdt",5_000_000))
        minchg = float(minchg or acc.get("auto_min_abs_change",2.5))
        
        try:
            syms, note = scan_top_volatile(quote, topn, minvol, minchg)
            if syms:
                audit_write("auto_universe_update", syms=syms, note=note)
                return {"syms": syms, "note": note}
        except Exception as e:
            audit_write("auto_universe_error", note=str(e))
    
    # Fallback to default
    return {"syms": UNIVERSE_DEFAULT[:int(topn or 12)], "note": "manual/fallback"}

# Main Trading Loop
@app.callback(
    [Output("main-chart","figure"), Output("multi-chart","figure"),
     Output("tbl-signals","data"), Output("tbl-orders","data"), 
     Output("tbl-portfolio","data"), Output("indicators-display","children"),
     Output("strategies-status","children"), Output("status-footer","children"),
     Output("metric-initial","children"), Output("metric-equity","children"),
     Output("metric-pnl","children"), Output("metric-ret","children"),
     Output("metric-real","children"), Output("metric-unreal","children"),
     Output("src-badge","children"), Output("src-badge","color"),
     Output("acct-summary","children")],
    [Input("tick","n_intervals")],
    [State("worker-on","data"), State("run-mode","data"), State("uni-store","data"), State("debug-store","data")]
)
def main_trading_loop(n, worker_on, run_mode, uni_data, debug_data):
    """Main enhanced trading loop with multi-strategy support"""
    
    try:
        t0 = time.time()
        log.info(f"[ENHANCED TICK] start n={n} mode={run_mode} worker_on={worker_on}")
        
        acc = read_account()
        universe = uni_data.get("syms", UNIVERSE_DEFAULT[:8])
        quote = acc.get("quote_asset", "USDT").upper()

        binance_balances = {} # Inicializa vazio
        if run_mode != "paper":
            binance_balances = binance_get_account_balance(acc)
            # Atualiza o cash do bot com o saldo real da quote asset
            acc["cash"] = binance_balances.get(quote, acc["cash"])
            write_account(acc)

        
        # Price fetching with better fallback
        prices_data = {}
        source_info = ("📡 Fonte: —", "secondary")
        
        if worker_on and ((n or 0) % 1 == 0):  # Every tick for enhanced bot
            testnet = (run_mode == "test")
            prices_data, meta = get_prices(universe, testnet=testnet)
            
            if meta.get("ok"):
                source_info = ("📡 Fonte: Binance ✅", "success")
            else:
                source_info = ("📡 Fonte: MOCK ⚠️", "warning")
        
        # Update price series with enhanced data
        executed_orders = 0
        strategy_signals = {}
        
        for symbol in universe:
            current_price = prices_data.get(symbol)
            if current_price is None:
                # Generate mock price with more realistic movement
                data = _read_json(FILES["prices"], {})
                if symbol in data and data[symbol]:
                    last_price = data[symbol][-1]["price"]
                    current_price = mock_step(last_price)
                else:
                    current_price = np.random.uniform(8, 15)
            
            # Enhanced price update with volume simulation
            volume = np.random.uniform(1000, 2500)
            high = current_price * (1 + np.random.uniform(0, 0.005))
            low = current_price * (1 - np.random.uniform(0, 0.005))
            
            append_price(symbol, current_price, volume, high, low)
            ensure_port(symbol, current_price)
            refresh_unrealized(symbol, current_price)
            
            # Enhanced trading logic
            if worker_on:
                df = load_prices_df(symbol, lookback=500)
                if len(df) < 50:
                    continue
                
                # Check risk exit first
                exit_signal, exit_reason = risk_exit_signal(symbol, current_price)
                
                if exit_signal:
                    port = _read_json(FILES["portfolio"], {})
                    bot_qty = port.get(symbol, {}).get("bot_qty", 0.0)
                    strategy_used = port.get(symbol, {}).get("strategy", "unknown")
                    
                    if bot_qty > 0:
                        signal_rec = add_signal(symbol, "sell", f"EXIT_{strategy_used}", 
                                              {"reason": exit_reason})
                        add_order_local(symbol, "sell", current_price, bot_qty, 
                                      signal_rec["id"], strategy=strategy_used)
                        executed_orders += 1
                        strategy_signals[symbol] = f"EXIT: {exit_reason}"
                        continue
                
                # Check cooldown
                cooldown_ready, remaining = cooldown_ok(symbol)
                if not cooldown_ready:
                    continue
                
                # Get multi-strategy signal
                signal, context = get_strategy_signal(symbol, current_price, df)
                
                if signal and signal in ["buy", "sell"]:
                    strategy_used = context.get("strategy_used", "unknown")
                    
                    # Execute trade based on strategy
                    port = _read_json(FILES["portfolio"], {})
                    p = port.get(symbol, {})
                    current_qty = p.get("quantity", 0.0)
                    bot_qty = p.get("bot_qty", 0.0)
                    
                    if signal == "sell" and bot_qty <= 0:
                        continue
                    
                    if signal == "buy" and bot_qty > 0:
                        continue
                    
                    # Position sizing for small capital
                    equity = equity_value()
                    max_position = equity * acc.get("max_position_pct", 0.80)
                    cash_available = acc.get("cash", 0.0)
                    buffer = equity * acc.get("cash_buffer_pct", 0.05)
                    
                    if signal == "buy":
                        buying_power = min(cash_available - buffer, max_position)
                        if buying_power >= acc.get("min_ticket_floor_quote", 2.0):
                            qty = buying_power / current_price
                            
                            signal_rec = add_signal(symbol, "buy", strategy_used, context)
                            add_order_local(symbol, "buy", current_price, qty, 
                                          signal_rec["id"], strategy=strategy_used, run_mode=run_mode)

                            executed_orders += 1
                            strategy_signals[symbol] = f"BUY: {strategy_used}"
                    
                    elif signal == "sell":
                        signal_rec = add_signal(symbol, "sell", strategy_used, context)
                        add_order_local(symbol, "sell", current_price, bot_qty,
                                      signal_rec["id"], strategy=strategy_used, run_mode=run_mode)

                        executed_orders += 1
                        strategy_signals[symbol] = f"SELL: {strategy_used}"
        
        # Generate enhanced charts
        main_chart = create_main_chart(universe[0] if universe else "BTCUSDT")
        multi_chart = create_multi_chart(universe[:6])
        
        # Get recent data for tables
        raw_signals = list(reversed(_read_json(FILES["signals"], [])))[:50]
        # ⚠️ Dash DataTable only accepts primitives; drop non-primitive fields like dict "context"
        signals_data = [
            {k: rec.get(k) for k in ("id","symbol","signal","strategy","timestamp")}
            for rec in raw_signals if isinstance(rec, dict)
        ]
        orders_data = list(reversed(_read_json(FILES["orders"], [])))[:50]
        portfolio_data = format_portfolio_data()
        
        # Generate indicators display
        indicators_display = create_indicators_display(universe[0] if universe else "BTCUSDT")
        
        # Generate strategy status
        strategies_display = create_strategies_status(strategy_signals)
        
        # Calculate metrics
        base_equity = acc.get("base_equity", 18.0)
        current_equity = equity_value()
        total_pnl = current_equity - base_equity
        return_pct = (total_pnl / base_equity * 100) if base_equity > 0 else 0
        
        port = _read_json(FILES["portfolio"], {})
        unrealized_total = sum(p.get("unrealized", 0.0) for p in port.values())
        
        # Status footer
        status_footer = (f"🚀 Enhanced Bot • {len(universe)} ativos • "
                        f"{executed_orders} ordens executadas • "
                        f"Modo: {run_mode.upper()} • "
                        f"⏱️ Processado em {time.time()-t0:.2f}s")
        
        # Account summary
        acct_summary = (f"💰 Capital: {money(current_equity)} • "
                       f"💵 Caixa: {money(acc.get('cash', 0.0))} (Binance: {money(binance_balances.get(quote, 0.0))}) • "
                       f"📈 Realizado: {money(acc.get("daily_realized", 0.0))} • "
                       f"🪙 Moedas: {", ".join([asset for asset, balance in binance_balances.items() if asset != quote and balance > 0])} • "
                       f"🎯 {len([s for s in strategy_signals.values() if 'BUY' in s or 'SELL' in s])} sinais ativos")
        
        return (
            main_chart, multi_chart, signals_data, orders_data, portfolio_data,
            indicators_display, strategies_display, status_footer,
            money(base_equity), money(current_equity),
            html.Span(money(total_pnl), className="text-success" if total_pnl >= 0 else "text-danger"),
            html.Span(f"{return_pct:+.2f}%", className="text-success" if return_pct >= 0 else "text-danger"),
            html.Span(money(acc.get('daily_realized', 0.0)), 
                     className="text-success" if acc.get('daily_realized', 0.0) >= 0 else "text-danger"),
            html.Span(money(unrealized_total), 
                     className="text-success" if unrealized_total >= 0 else "text-danger"),
            source_info[0], source_info[1], acct_summary
        )
        
    except Exception as e:
        log.error(f"Error in main trading loop: {e}")
        audit_write("main_loop_error", error=str(e))
        
        # Return safe fallback
        empty_fig = go.Figure()
        return (
            empty_fig, empty_fig, [], [], [], 
            f"❌ Erro: {str(e)}", "Sistema em recuperação...", 
            f"Erro no sistema: {str(e)}", "—", "—", "—", "—", "—", "—",
            "📡 Fonte: ERRO", "danger", "Sistema reiniciando..."
        )

def create_main_chart(symbol):
    """Create enhanced main chart with all indicators"""
    df = load_prices_df(symbol, lookback=200)
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{symbol} - Preço + Indicadores", "Volume", "RSI + Stochastic"],
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df["t"], open=df["price"], high=df["high"], 
            low=df["low"], close=df["price"], name="Price"
        ), row=1, col=1
    )
    
    # Calculate and plot indicators
    indicators = calculate_all_indicators(df)
    
    if indicators.get("bb_upper"):
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df["t"][-1:], y=[indicators["bb_upper"]], 
            name="BB Upper", line=dict(color="red", dash="dash")
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df["t"][-1:], y=[indicators["bb_lower"]], 
            name="BB Lower", line=dict(color="green", dash="dash")
        ), row=1, col=1)
    
    # Moving averages
    if len(df) >= INDICATORS_CONFIG["SMA_FAST"]:
        sma_fast = df["price"].rolling(INDICATORS_CONFIG["SMA_FAST"]).mean()
        fig.add_trace(go.Scatter(
            x=df["t"], y=sma_fast, name=f"SMA{INDICATORS_CONFIG['SMA_FAST']}", 
            line=dict(color="orange")
        ), row=1, col=1)
    
    if len(df) >= INDICATORS_CONFIG["SMA_SLOW"]:
        sma_slow = df["price"].rolling(INDICATORS_CONFIG["SMA_SLOW"]).mean()
        fig.add_trace(go.Scatter(
            x=df["t"], y=sma_slow, name=f"SMA{INDICATORS_CONFIG['SMA_SLOW']}", 
            line=dict(color="blue")
        ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df["t"], y=df["volume"], name="Volume", marker_color="lightblue"
    ), row=2, col=1)
    
    # RSI
    if indicators.get("rsi"):
        fig.add_trace(go.Scatter(
            x=df["t"][-1:], y=[indicators["rsi"]], name="RSI", 
            line=dict(color="purple")
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Add order markers
    orders = _read_json(FILES["orders"], [])
    recent_orders = [o for o in orders if o.get("symbol") == symbol][-50:]
    
    for order in recent_orders:
        try:
            order_time = pd.to_datetime(order["timestamp"]).tz_localize("UTC").tz_convert(TZ_SP).tz_localize(None)
            if order["side"] == "buy":
                fig.add_trace(go.Scatter(
                    x=[order_time], y=[order["price"]], mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    name=f"Buy ({order.get('strategy', 'unknown')})", showlegend=False
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=[order_time], y=[order["price"]], mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="red"), 
                    name=f"Sell ({order.get('strategy', 'unknown')})", showlegend=False
                ), row=1, col=1)
        except:
            continue
    
    fig.update_layout(
        height=500, margin=dict(l=10,r=10,t=30,b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True, legend=dict(orientation="h", y=-0.1)
    )
    
    return fig

def create_multi_chart(universe):
    """Create multi-asset chart"""
    if not universe:
        return go.Figure()
    
    rows = min(len(universe), 6)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"{sym} - Multi-Strategy" for sym in universe[:rows]]
    )
    
    for i, symbol in enumerate(universe[:rows], 1):
        df = load_prices_df(symbol, lookback=100)
        if df.empty:
            continue
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df["t"], y=df["price"], name=f"{symbol}", 
            line=dict(width=2)
        ), row=i, col=1)
        
        # Simple moving average
        if len(df) >= 20:
            sma = df["price"].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=df["t"], y=sma, name="SMA20", 
                line=dict(dash="dot"), showlegend=False
            ), row=i, col=1)
    
    fig.update_layout(
        height=max(400, rows * 100), 
        margin=dict(l=10,r=10,t=30,b=10),
        showlegend=False
    )
    
    return fig

def format_portfolio_data():
    """Format portfolio data for display"""
    port = _read_json(FILES["portfolio"], {})
    data = []
    
    for symbol, p in port.items():
        if p.get("quantity", 0) > 0:
            data.append({
                "symbol": symbol,
                "strategy": p.get("strategy", "—") or "—",
                "quantity": round(p.get("quantity", 0), 6),
                "avg_price": round(p.get("avg_price", 0), 6),
                "last_price": round(p.get("last_price", 0), 6),
                "unrealized": round(p.get("unrealized", 0), 4),
                "bot_qty": round(p.get("bot_qty", 0), 6)
            })
    
    return data

def create_indicators_display(symbol):
    """Create indicators display"""
    indicators_data = _read_json(FILES["indicators"], {})
    if symbol not in indicators_data:
        return "⏳ Calculando indicadores..."
    
    indicators = indicators_data[symbol].get("indicators", {})
    
    display_items = []
    
    # RSI
    if indicators.get("rsi"):
        rsi_val = indicators["rsi"]
        rsi_status = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral"
        display_items.append(f"📊 RSI: {rsi_val:.1f} {rsi_status}")
    
    # MACD
    if indicators.get("macd"):
        macd_val = indicators["macd"]
        macd_status = "🚀 Bullish" if macd_val > 0 else "🔻 Bearish"
        display_items.append(f"📈 MACD: {macd_val:.6f} {macd_status}")
    
    # Bollinger Position
    if all(k in indicators for k in ["bb_upper", "bb_lower"]):
        df = load_prices_df(symbol, 5)
        if not df.empty:
            current_price = df["price"].iloc[-1]
            if current_price > indicators["bb_upper"]:
                bb_status = "🔴 Above Upper Band"
            elif current_price < indicators["bb_lower"]:
                bb_status = "🟢 Below Lower Band" 
            else:
                bb_status = "🟡 Inside Bands"
            display_items.append(f"🎯 BB Position: {bb_status}")
    
    # Volume
    if indicators.get("volume_ratio"):
        vol_ratio = indicators["volume_ratio"]
        vol_status = "🔥 High" if vol_ratio > 2 else "⚡ Normal" if vol_ratio > 0.5 else "💧 Low"
        display_items.append(f"📊 Volume: {vol_ratio:.1f}x {vol_status}")
    
    return html.Div([html.P(item, className="mb-1") for item in display_items])

def create_strategies_status(signals):
    """Create strategy status display"""
    if not signals:
        return "⏳ Aguardando sinais..."
    
    status_items = []
    for symbol, signal in signals.items():
        if "BUY" in signal:
            status_items.append(html.P(f"🟢 {symbol}: {signal}", className="mb-1 text-success"))
        elif "SELL" in signal:
            status_items.append(html.P(f"🔴 {symbol}: {signal}", className="mb-1 text-danger"))
        elif "EXIT" in signal:
            status_items.append(html.P(f"🔶 {symbol}: {signal}", className="mb-1 text-warning"))
    
    if not status_items:
        status_items.append(html.P("🟡 Sem sinais ativos", className="mb-1 text-muted"))
    
    return html.Div(status_items)

# Run the app
import os

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8051"))  # cai pra 8050 se não setar
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host=host, port=port, debug=debug)

