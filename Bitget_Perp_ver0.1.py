#-*-coding:utf-8-*-
from pickle import TRUE
import subprocess
import time
import datetime
#from datetime import datetime, timezone, timedelta 
import requests
import json
import ccxt
from telegram import Bot
import asyncio
import random
import hashlib
import hmac
from urllib.parse import urlparse
import numpy as np
from pandas import DataFrame
import sqlite3
#import dbinsert
import myutil2
import base64, hashlib, hmac, json, requests, time
import sys
import pandas as pd
import client
import math
import pprint
import bitget.mix.market_api as market
import bitget.mix.account_api as accounts
import bitget.mix.position_api as position
import bitget.mix.order_api as order
import bitget.mix.plan_api as plan
import bitget.mix.trace_api as trace
import bitget.mix.plan_api as plan
import bitget.spot.wallet_api as wallet
import json
import os
from dotenv import load_dotenv
import math
#from currency_converter import CurrencyConverter
import threading

# Load environment variables from .env file
load_dotenv()


position_side = sys.argv[1]
account = sys.argv[2]
coin = sys.argv[3]

class CurrencyConverter:
    def __init__(self, url=None):
        self.rates = {'USD': 1.0, 'KRW': 1300.0}  # Default rates (USD to KRW)
    
    def convert(self, amount, from_curr, to_curr):
        """Convert between currencies using fixed rates"""
        if from_curr == to_curr:
            return amount
        try:
            # Try to fetch from an API if needed
            if from_curr == 'USD' and to_curr == 'KRW':
                # Using a default rate - in production, update this from a real API
                rate = self.rates.get('KRW', 1300.0)
                return amount * rate
        except:
            pass
        return amount  # Return original amount if conversion fails

# telegram info
# Prefer environment variables; fallback to existing values for compatibility
my_token = '744382085:AAE-4OscklWp-YC7IhziQUH3djVun0IvIQY'
chat_id = '731080199'
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', my_token)
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', chat_id)
bot = Bot(token=TOKEN)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.302 Safari/537.36'}


def tg_send(text):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": str(text)}, timeout=10)
    except Exception as e:
        # Avoid crashing trading logic if Telegram fails
        print('Telegram send failed:', type(e).__name__, e)

def exit_alarm_enable(avg_price,close_price,side):
    if close_price != avg_price and avg_price != None:
        if side == "long":
            if avg_price !=0:
                profit=(close_price - avg_price)/avg_price*100
                return profit
            else:
                return 0
        elif side == "short":
            if avg_price !=0:
                profit=(avg_price - close_price)/avg_price*100
                return profit
            else:
                return 0
    else:
        return 0

def read_json(filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename) as f:
        try:
            live24data = json.load(f)
        except:
            return 0
    lock.release()
    return live24data

def get_pos_index(live24, symbol, position_side):
    for i in range(len(live24['data'])):
        #print("get_pos_index>>i:{}/symbol:{}/holdSide:{}".format(i, live24['data'][i]['symbol'], live24['data'][i]['holdSide']))
        if live24['data'][i]['symbol'] == symbol and live24['data'][i]['holdSide'] == position_side:
            return i
    # Raise an exception if no matching index is found
    raise ValueError(f"No matching position found for symbol: {symbol}, position_side: {position_side}")

def return_true_after_minutes(minute,timestamp):
    target_timestamp = timestamp + (minute * 60)  # n분 후의 타임스탬프 계산
    if time.time() > target_timestamp:
        ret=1
        diff= target_timestamp-time.time()
    else:
        ret=0
        diff= target_timestamp-time.time()
    return ret,diff


def short_action(action):
    """
    Unified function for short entry/exit.
    action: 'entry' or 'exit'
    """
    if action == 'entry':
        print("entry>>liquidationPrice*0.9:{}<close:{}".format(liquidationPrice*0.9, close_price))
        sign = 1
        msg_prefix = 'entry>>'
        msg_mid = '+'
        msg_usd = '+1USD'
    elif action == 'exit':
        sign = -1
        msg_prefix = 'exit>>'
        msg_mid = '-'
        msg_usd = '-1USD'
    else:
        raise ValueError("action must be 'entry' or 'exit'")

    percents = [0.08, 0.04, 0.02, 0.01]
    percent_labels = ['8%', '4%', '2%', '1%']
    for i, pct in enumerate(percents):
        try:
            cal_amount = round(margin * pct)
            message = f"{msg_prefix}[{account}][{position_side}]liquidationPrice*0.9:{{}}<close_price:{{}}{msg_mid}{{}}:{{}}".format(
                liquidationPrice*0.9, close_price, percent_labels[i], cal_amount)
            print(message)
            result = accountApi.margin(symbol, marginCoin='USDT', productType='USDT-FUTURES', amount=cal_amount*sign, holdSide=position_side)
            if action == 'exit':
                print(result)
            #tg_send(message)
            break
        except Exception:
            continue
    else:
        try:
            message = f"{msg_prefix}[{account}][{position_side}]liquidationPrice*0.9:{{}}<close_price:{{}}{msg_usd}".format(liquidationPrice*0.9, close_price)
            result = accountApi.margin(symbol, marginCoin='USDT', productType='USDT-FUTURES', amount=1*sign, holdSide=position_side)
            print(message)
            if action == 'exit':
                print(result)
            #tg_send(message)
        except Exception:
            pass


def long_action(action):
    """
    Unified function for long entry/exit.
    action: 'entry' or 'exit'
    """
    if action == 'entry':
        print("entry>>liquidationPrice:{}*1.1:{}>close:{}".format(liquidationPrice, liquidationPrice * 1.1, close_price))
        sign = 1
        msg_prefix = 'entry>>'
        msg_mid = '+'
        msg_usd = '+1USD'
    elif action == 'exit':
        print("exit>>liquidationPrice:{}*1.1:{}<close:{}".format(liquidationPrice, liquidationPrice * 1.1, close_price))
        sign = -1
        msg_prefix = 'exit>>'
        msg_mid = '-'
        msg_usd = '-1USD'
    else:
        raise ValueError("action must be 'entry' or 'exit'")

    percents = [0.08, 0.04, 0.02, 0.01]
    percent_labels = ['8%', '4%', '2%', '1%']
    for i, pct in enumerate(percents):
        try:
            cal_amount = round(margin * pct)
            message = f"{msg_prefix}[{account}][{position_side}]liquidationPrice*1.1:{{}}>close_price:{{}}{msg_mid}{{}}:{{}}".format(
                liquidationPrice * 1.1, close_price, percent_labels[i], cal_amount)
            print(message)
            result = accountApi.margin(symbol, marginCoin='USDT', productType='USDT-FUTURES', amount=cal_amount * sign, holdSide=position_side)
            if action == 'exit':
                print(result)
            # tg_send(message)
            break
        except Exception:
            continue
    else:
        try:
            message = f"{msg_prefix}[{account}][{position_side}]liquidationPrice*1.1:{{}}>close_price:{{}}{msg_usd}".format(liquidationPrice * 1.1, close_price)
            result = accountApi.margin(symbol, marginCoin='USDT', productType='USDT-FUTURES', amount=1 * sign, holdSide=position_side)
            print(message)
            if action == 'exit':
                print(result)
            # tg_send(message)
        except Exception:
            pass

def short_entry():
    short_action('entry')

def short_exit():
    short_action('exit')

def long_entry():
    long_action('entry')

def long_exit():
    long_action('exit')

def get_exchange_credentials(account):
    """계정별 API 키를 환경변수에서 로드"""
    account_upper = account.upper()
    
    api_key = os.getenv(f'{account_upper}_API_KEY')
    secret_key = os.getenv(f'{account_upper}_SECRET_KEY')
    passphrase = os.getenv(f'{account_upper}_PASSPHRASE')
    
    if not all([api_key, secret_key, passphrase]):
        raise RuntimeError("Exchange credentials not configured")
    
    return api_key, secret_key, passphrase

def is_night_time_utc9():
    # UTC+9 (KST)
    kst = timezone(timedelta(hours=9)) 
    now = datetime.now(kst).time()
    # 21:00 ~ 24:00 OR 00:00 ~ 09:00
    return now >= datetime.strptime("21:00","%H:%M").time() or \
           now < datetime.strptime("09:00", "%H:%M").time()

def is_even_hour(): 
    hour = datetime.now().hour
    return hour % 2 == 0

def is_weekend_risk_zone(timestamp_ms):
    dt = datetime.datetime.utcfromtimestamp(timestamp_ms / 1000)
    weekday = dt.weekday()
    hour = dt.hour

    # 토요일 전체
    if weekday == 5:
        return True

    # 일요일 + 저유동성 구간
    if weekday == 6 and hour < 22:
        return True

    return False

def resolve_market_state(asset, trend_strength, timestamp, is_macro_event=False):
    if asset == "BTCUSDT":
        return resolve_btc_state(trend_strength, timestamp, is_macro_event)
    elif asset == "QQQUSDT":
        return resolve_qqq_state(trend_strength, is_macro_event)
    else:
        return "normal"

def resolve_btc_state(trend_strength, timestamp, is_macro_event):
    if is_macro_event:
        return "post_event"

    if trend_strength > 0.6:
        return "trend_up"

    if is_weekend_risk_zone(timestamp):
        return "weekend_range"

    return "weekday_swing"

def resolve_qqq_state(trend_strength, is_macro_event):
    if is_macro_event:
        return "macro_event"

    if trend_strength > 0.6:
        return "trend"

    return "normal"

def trend_strength(df, atr, span=21):
    """
    추세 강도 = EMA 기울기 / ATR
    """
    if len(df) < span + 2 or atr == 0:
        return 0.0

    ema = df["close"].ewm(span=span, adjust=False).mean()

    # 최근 기울기
    slope = ema.iloc[-1] - ema.iloc[-2]

    # ATR로 정규화
    strength = slope / atr
    return strength
    
def calculate_atr(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) <= period:
        return None

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    return atr.iloc[-1]

def get_alpha(symbol, state):
    #alpha_key = STATE_TO_ALPHA_KEY[symbol][state]
    return ALPHA_TABLE[symbol][state]["alpha"]

def gap_velocity(gap_hist, window=3):
    """
    gap_hist: list or deque of gap values (float)
    window: 최근 몇 개 기준으로 속도 계산
    """
    if len(gap_hist) < window + 1:
        return 0.0

    diffs = [
        gap_hist[i] - gap_hist[i - 1]
        for i in range(-window + 1, 0)
    ]
    return sum(diffs) / len(diffs)

def compute_T_control(
    T_base,
    gap_v,          # gap velocity (확장 속도)
    hedge_pnl,      # 현재 hedge pnl (음수면 손실)
    free_margin,
    used_margin,
    alpha,          # 현재 시장 변동성 기반 alpha
    alpha_base,     # 기준 alpha
    T_min,
    T_max,
    gap_scale=1.0
):
    """
    T_control:
    - 다음 진입까지 기다릴 봉 수
    - 클수록 보수적, 작을수록 공격적
    """

    # 1️⃣ 마진 스트레스
    if used_margin > 0:
        margin_stress = 1.0 - (free_margin / (free_margin + used_margin))
    else:
        margin_stress = 0.0

    # 2️⃣ 손실 스트레스 (손실일 때만 반영)
    pnl_stress = max(0.0, -hedge_pnl)

    # 3️⃣ 갭 확장 스트레스
    gap_stress = abs(gap_v) * gap_scale

    # 4️⃣ 시장 변동성 스트레스
    vol_stress = max(0.0, alpha / alpha_base - 1.0)

    # 5️⃣ 총 스트레스 (선형 합)
    stress = (
        0.4 * margin_stress +
        0.3 * pnl_stress +
        0.2 * gap_stress +
        0.1 * vol_stress
    )

    # 6️⃣ T 조정
    T = T_base * (1.0 + stress)

    # 7️⃣ 하드 클램프
    T = max(T_min, min(T, T_max))

    return int(round(T))

def calc_exit_levels(
    price: float,
    atr: float,
    remaining_count: int,
    base_profit: float = 0.01,      # 1% 최소 이익
    vol_min: float = 0.5,
    vol_max: float = 2.0
):
    """
    가변 Exit Interval 계산 함수

    :param price: 현재 가격
    :param atr: ATR 값 (같은 타임프레임)
    :param remaining_count: 남은 exit count (기타줄 수)
    :param base_profit: 최소 이익 비율 (0.01 = 1%)
    :param vol_min: 변동성 하한 clamp
    :param vol_max: 변동성 상한 clamp
    :return: exit_levels (list of profit ratios)
    """

    if remaining_count <= 0:
        return []

    # 1️⃣ 상대 변동성 계산
    raw_vol_factor = atr / price

    # 2️⃣ 변동성 clamp
    vol_factor = max(vol_min, min(raw_vol_factor, vol_max))

    # 3️⃣ 이번 사이클 최대 이익 목표
    max_profit = base_profit * vol_factor

    # 4️⃣ exit level 생성 (base → max)
    exit_levels = np.linspace(
        base_profit,
        max_profit,
        remaining_count
    )

    return exit_levels.tolist()

def cycle_entry_filter_hysteresis(
    side: str,
    profit: float,
    adjusted_alpha: float,
    trend_strength: float,
    enter_eps: float = 0.05,
    exit_eps: float = 0.02,
    in_position: bool = False,
) -> bool:
    """
    진입/유지 히스테리시스 포함 필터
    """

    if profit <= adjusted_alpha:
        return False

    if side == "long":
        threshold = exit_eps if in_position else enter_eps
        return trend_strength > threshold

    elif side == "short":
        threshold = -exit_eps if in_position else -enter_eps
        return trend_strength < threshold

    else:
        raise ValueError("side must be 'long' or 'short'")

def calc_APAE(
    prev_avg_price: float,
    curr_avg_price: float,
    prev_price: float,
    curr_price: float,
    side: str,
    eps: float = 1e-9
) -> float:
    """
    APAE (Average Price Adjustment Efficiency) 계산

    APAE = 평단이 유리하게 이동한 거리 / 가격이 이동한 전체 거리

    :param prev_avg_price: 이전 평단
    :param curr_avg_price: 현재 평단
    :param prev_price: 이전 기준 가격 (예: 직전 캔들 close)
    :param curr_price: 현재 가격
    :param side: 'long' 또는 'short'
    :param eps: 0 division 방지용
    :return: APAE 값 (0 ~ 1 이상도 가능)
    """

    # 가격 이동 거리
    price_move = abs(curr_price - prev_price)

    if price_move < eps:
        return 0.0

    if side == 'long':
        # 롱: 평단이 내려가면 유리
        avg_move = prev_avg_price - curr_avg_price
    elif side == 'short':
        # 숏: 평단이 올라가면 유리
        avg_move = curr_avg_price - prev_avg_price
    else:
        raise ValueError("side must be 'long' or 'short'")

    # 불리한 방향 이동은 0 처리
    avg_move = max(avg_move, 0.0)

    apae = avg_move / price_move
    return round(apae, 4)

def can_rebuild_position(
    side,
    current_price,
    average_price,
    anchor_price,
    initial_step,
    increase_unit,
    current_count,
    trend_strength,
    trend_eps
):
    """
    부분청산 후 앵커 기준 재확장 가능 여부 판단
    """

    if anchor_price is None or current_count <= 0:
        return False

    # 다음 회차 기준 적정 레벨 계산
    next_count = current_count + 1
    entry_price = get_next_entry_level(
        anchor_price,
        side,
        initial_step,
        increase_unit,
        next_count
    )

    # 1️⃣ 추세 필터
    if side == "long" and trend_strength <= trend_eps:
        return False
    if side == "short" and trend_strength >= -trend_eps:
        return False

    # 2️⃣ 현재 수익 상태
    if side == "long" and current_price <= average_price:
        return False
    if side == "short" and current_price >= average_price:
        return False

    # 3️⃣ 평단 갭 확장 여부
    if side == "long":
        expands_gap = entry_price < average_price
    else:
        expands_gap = entry_price > average_price

    return expands_gap

def get_next_entry_level(
    base_price,
    side,
    ankor_step,
    increase_unit,
    count
):
    """
    다음 회차 진입 가격 계산 (증분 방식)
    """

    if count <= 0:
        return base_price

    next_gap = ankor_step + (count - 1) * increase_unit

    if side == "short":
        result_price = base_price * (1 + next_gap)
    elif side == "long":
        result_price = base_price * (1 - next_gap)
    else:
        raise ValueError("side는 'short' 또는 'long'")

    return round(result_price, 4)

if __name__ == "__main__":
    cnt=0
    cntm=0
    minute=0
    minutem=0
    pre_short_count=0
    pre_long_count=0
    pre_set_lev = 0
    long_flag = False
    short_flag = False
    productType = 'USDT-FUTURES'
    marginC = 'USDT'
    productT='umcbl'
    profit_base_rate = 0.01 # 1% tp 세팅
    short_profit_line = 0.99 # 기타줄 초기 레벨
    short_profit_line_adjust = 0.999 # 1% 넘었을때 close_price 기준으로 세팅갭 0.1%
    long_profit_line = 1.01 # 기타줄 초기 레벨
    long_profit_line_adjust = 1.011 # 1% 넘었을때 close_price 기준으로 세팅갭 0.1%


    if coin == 'QQQUSDT':
        symbol = 'QQQUSDT'
        symbol2 = 'qqqusdt'
        #entry_percent = 0.5 # 최초 진입점 및, 이후 진입 기준점 
        bet_sizex_div = 0.5 # 진입비율 계산용 계수 
        bet_size_base = 0.01 # 최소 진입 사이즈 
        gap_base_rate = 0.001 # 갭 계산용 계수
        gap_expend_rate = 0.0001 # 갭 확장 계산용 계수
        T_PARAMS = dict( T_base=8, T_min=3, T_max=16, alpha_base=0.0025, # 0.2~0.3% 
        gap_scale=1.5 )

    elif coin == 'BTCUSDT':
        symbol = 'BTCUSDT'
        symbol2 = 'btcusdt'
        #entry_percent = 0.5
        bet_sizex_div = 0.005
        bet_size_base = 0.0001
        gap_base_rate = 0.002
        gap_expend_rate = 0.0001
        T_PARAMS = dict( T_base=6, T_min=2, T_max=12, alpha_base=0.008, # 0.5~1% 영역 
        gap_scale=1.0 )

    ALPHA_TABLE = {
        "BTCUSDT": {
            "trend_up": {
                "alpha": 0.012,
                "desc": "강한 상승 추세"
            },
            "weekday_swing": {
                "alpha": 0.008,
                "desc": "평일 스윙"
            },
            "weekend_range": {
                "alpha": 0.004,
                "desc": "주말 횡보"
            },
            "post_event": {
                "alpha": 0.015,
                "desc": "이벤트 직후"
            }
        },
    "QQQUSDT": {
            "macro_event": {
                "alpha": 0.010,   # 1.0% — 이벤트 직후, 변동성 폭주 구간
                "desc": "매크로 이벤트 직후"
            },
            "trend": {
                "alpha": 0.006,   # 0.6% — 명확한 추세, 눌림/확장 대응
                "desc": "추세 구간"
            },
            "normal": {
                "alpha": 0.003,   # 0.3% — 평시 레인지
                "desc": "일반 구간"
            }
        }
    }

    filename2  = coin+'_'+account+'.json'
    filename  = account+'.json'

    # Load API credentials from environment variables
    api_key, secret_key, passphrase = get_exchange_credentials(account)
    
    # Initialize ccxt client
    c = ccxt.bitget({'apiKey' : api_key,'secret' : secret_key, 'password' : passphrase, 'options':{'defaultType':'swap'}})

    marketApi = market.MarketApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    positionApi = position.PositionApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    orderApi = order.OrderApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    accountApi = accounts.AccountApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    planApi = plan.PlanApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    indexname = account+'_'+coin+'_'+position_side

    # Load Main account credentials for wallet operations
    main_api_key, main_secret_key, main_passphrase = get_exchange_credentials('Main')
    walletApi = wallet.WalletApi(main_api_key, main_secret_key, main_passphrase, use_server_time=False, first=False)

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    while True:
        close_price = float(marketApi.ticker(symbol,'USDT-FUTURES')['data'][0]['lastPr'])
        # 1. 데이터 가져오기 (충분한 데이터 확보를 위해 limit을 100 이상 권장)
        raw_data = marketApi.get_perp_candles("BTCUSDT", "1H", limit=100)
        data =raw_data['data']
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume"
        ]
        df = pd.DataFrame(data, columns=cols)
        atr = calculate_atr(df, period=14)

        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = df[c].astype(float)

        df["open_time"] = df["open_time"].astype("int64")
        T_market = trend_strength(df, atr, span=21)
        price = df["close"].iloc[-1]
        print(f"현재 1시간봉 ATR/close_price: {atr/price}") 
        print("T_market Trend Strength:", T_market) 

        hedge_pnl = 0.2 
        free_margin = 500
        used_margin = 1500

        alpha = atr/price
        
        gap_hist = [0.012, 0.014] # 예시
        
        gap_v = gap_velocity(gap_hist)

        #print( 'T_base:', T_PARAMS['T_base'], type(T_PARAMS['T_base']), 'gap_v:', gap_v, type(gap_v), 'hedge_pnl:', hedge_pnl, type(hedge_pnl), 'free_margin:', free_margin, type(free_margin), 'used_margin:', used_margin, type(used_margin), 'alpha:', alpha, type(alpha), 'alpha_base:', T_PARAMS['alpha_base'], type(T_PARAMS['alpha_base']), 'T_min:', T_PARAMS['T_min'], type(T_PARAMS['T_min']), 'T_max:', T_PARAMS['T_max'], type(T_PARAMS['T_max']), 'gap_scale:', T_PARAMS['gap_scale'], type(T_PARAMS['gap_scale']) )
        T_control = compute_T_control( T_PARAMS['T_base'], gap_v, hedge_pnl, free_margin, used_margin, alpha, T_PARAMS['alpha_base'], T_PARAMS['T_min'], T_PARAMS['T_max'], T_PARAMS['gap_scale'] )
        print("T_control (bars):", T_control)
        base_time = 12
        exit_interval = base_time * (T_control / abs(T_market))

        # --------------------------------------------------
        # t_market / t_control concept (future use)
        #
        # t_market  : estimated market speed (ATR, gap_velocity, trend)
        # t_control : system reaction speed (risk tolerance, phase control)
        #
        # Idea:
        # - Use (t_market / t_control) to adjust gap size
        # - Use (t_control / t_market) to adjust entry timing
        #
        # Purpose:
        # - Prevent overreaction in fast markets
        # - Slow down entries under stress
        # - Keep avg reference intact while improving resilience
        #
        # Not activated yet (phase1 uses fixed gap logic)
        # --------------------------------------------------

        # Gap control logic:
        # - Start with conservative base gap (0.2%)
        # - Expand gap gradually per entry until market-allowed cap (~0.38%)
        # - After cap is reached, gap is frozen
        # - Further risk control must shift from gap -> size differential (stress list)


        state = resolve_market_state(
            asset=symbol,
            trend_strength=T_market,
            timestamp=time.time(),
            is_macro_event=False
        )

        alpha_scale = get_alpha(symbol, state)
        alpha = (atr / price) * alpha_scale

        print(f"Market State: {state}, Alpha Scale: {alpha_scale}, Adjusted Alpha*100: {alpha*100}%")
        print(f"Gap (price): {price * alpha}")

        chgUtc = float(marketApi.ticker(symbol,'USDT-FUTURES')['data'][0]['changeUtc24h'])*100
        chgUtcWoAbs = chgUtc
        balances = accountApi.account(coin,'USDT-FUTURES', marginCoin=marginC)
        free = float(balances['data']['available'])
        total = float(balances['data']['usdtEquity'])
        total_div4 = round(float(balances['data']['usdtEquity'])/2,1)  # temporary 이더 최소 분해능이 안나온다 테스트후 2->4 원복 예정
        amount=round(total/close_price,8)
        freeamount=round(free/close_price,8)
        cnt=cnt+1
        cntm=cntm+1
        live24data_condition = read_json(filename2)
        if live24data_condition:
            live24data = live24data_condition
            live24data_backup=live24data
        else:
            live24data=live24data_backup
        
        condition = read_json(filename)
        if condition:
            live24 = condition
            live24_backup=live24
        else:
            live24 = live24_backup

        print("exit_interval:", exit_interval)
        timeout = exit_interval # 9분마다 1% 기타줄 세팅 무식해..ㅋ 
        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')
        long_take_profit = live24data['long_take_profit'] #1.001 #live24data['long_take_profit']
        short_take_profit = live24data['short_take_profit'] #0.999 #live24data['short_take_profit']
        try:
            idx = get_pos_index(position,coin,position_side)
            if position_side == 'short':
                myutil2.live24flag('short_position_running',filename2,True)
            elif position_side == 'long':
                myutil2.live24flag('long_positon_running',filename2,True)
        except:
            # 모두 포지션 재개 조건 충족시 가장 작은 사이즈로 진입
            if position_side == 'short':
                print("short Positon not found/long_profit > alpha*100: {}/{}".format(long_profit, alpha*100))
                myutil2.live24flag('short_position_running',filename2,False)
                if long_profit > alpha*100 and T_market > 0.05: #can_enter_short and 0: #확실히 long 추세확인하고 진입 
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_profit_line,1), timeInForceValue='normal')
                    myutil2.live24flag('highest_short_price',filename2,float(close_price))
                    myutil2.live24flag('short_ankor_price',filename2,float(close_price))
                    message="[{}]1st Market Short Entry".format(account)
                    tg_send(message)
                    time.sleep(30)
            elif position_side == 'long':
                print("long Positon not found/short_profit > alpha*100: {}/{}".format(short_profit, alpha*100))
                myutil2.live24flag('long_position_running',filename2,False)
                if short_profit > alpha*100 and T_market < -0.05: #can_enter_long and 0: # 확실히 short 추세확인하고 진입
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='buy', tradeSide='open', marginMode='isolated', productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*long_profit_line,1), timeInForceValue='normal')
                    myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                    myutil2.live24flag('long_ankor_price',filename2,float(close_price))
                    message="[{}]1st Market Long Entry".format(account)
                    tg_send(message)
                    time.sleep(30)
            time.sleep(10)
            continue

        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')['data'][idx]
        liquidationPrice=round(float(position['liquidationPrice']),1)
        breakeven = round(float(position['breakEvenPrice']),1)

        unrealizedPnl=round(float(position['unrealizedPL']),1)
        #print(position)
        achievedProfits=round(float(position['achievedProfits']),1)
        avg_price = round(float(position['openPriceAvg']),3)

        short_profit = live24data['short_profit'] #1.001 #live24data['long_take_profit']
        long_profit = live24data['long_profit'] #0.999 #live24data['short_take_profit']

        if position_side == 'short':
            current_scale_index = live24data['sell_orders_count']
            ankor_price= live24data['short_ankor_price']
        elif position_side == 'long':
            current_scale_index = live24data['buy_orders_count']
            ankor_price = live24data['long_ankor_price']

        if account == 'Sub10':
            print("position_side: {}, close_price: {}, avg_price: {}, ankor_price: {}, gap_base_rate: {}, gap_expend_rate: {}, current_scale_index: {}, T_market: {}".format(position_side, close_price, avg_price, ankor_price, gap_base_rate, gap_expend_rate, current_scale_index, T_market))
            reentry_filter = can_rebuild_position(position_side,close_price,avg_price,ankor_price,float(gap_base_rate),float(gap_expend_rate),current_scale_index,T_market,0.02)
            entry_level = get_next_entry_level(avg_price,position_side,gap_base_rate,gap_expend_rate,current_scale_index + 1)
            #entry_level = get_entry_level(avg_price, position_side, float(gap_base_rate), float(gap_expend_rate),current_scale_index+1)
            if position_side == 'short':
                print("Short Reentry Filter: {}, Short Entry Level: {}".format(reentry_filter, entry_level))
            elif position_side == 'long':
                print("Long Reentry Filter: {}, Long Entry Level: {}".format(reentry_filter, entry_level))
        
        adjustment_count = current_scale_index + 1  # 1 ~ N

        exit_interval = exit_interval / adjustment_count
        MIN_EXIT_INTERVAL = 10  # bars or cycles
        exit_interval = round(max(exit_interval, MIN_EXIT_INTERVAL))

        # exit_interval_raw=1374
        # adjustment_count=22
        # exit_interval_final=53

        exit_levels = calc_exit_levels(
            price=price,
            atr=atr,
            remaining_count=current_scale_index
        )

        print(exit_levels)


 #       print("close_price:{}/avg_price:{}".format(close_price,avg_price))
        absamount = float(position['available'])

        short_gap = abs(close_price-live24data['short_avg_price'])
        long_gap = abs(close_price-live24data['long_avg_price'])

        leverage = float(position["leverage"])
        absamount_gap = abs(live24data['short_absamount']-live24data['long_absamount'])
        free_lev = float(freeamount) * float(leverage)
        short_lev = float(live24data['short_absamount']) * float(leverage)
        long_lev = float(live24data['long_absamount']) * float(leverage)
        #print("freeamount:{}/short:{}/long:{}/leverage:{}".format(freeamount,short_lev,long_lev,leverage))
        set_lev = float(absamount_gap) * float(leverage) /free_lev
        if set_lev > 50:
           set_lev = 50
        elif set_lev < 10:
           set_lev = 10
        set_lev = int(set_lev)
        #print("set_lev:{}".format(set_lev))
#        set_lev = 49
        if live24data['short_absamount'] > live24data['long_absamount']:
            if pre_set_lev != set_lev:
                message = "[{}][{}]set_leverage long {}x/short 10x".format(account,symbol2,set_lev)
                try:
                    result = accountApi.leverage_v3(symbol2,  productType = "USDT-FUTURES", marginCoin='USDT', longLeverage=set_lev, shortLeverage='10')
                except:
                    pass
#                tg_send(message)
                pre_set_lev = set_lev

        else:
            if pre_set_lev != set_lev:
                message = "[{}][{}]set_leverage short {}x/long 10x".format(account,symbol2,set_lev)
                try:
                    result = accountApi.leverage_v3(symbol2,  productType = "USDT-FUTURES", marginCoin='USDT', longLeverage='10', shortLeverage=set_lev)
                except:
                    pass
 #               tg_send(message)
                pre_set_lev = set_lev

        avg_break_gap= exit_alarm_enable(breakeven,avg_price,position_side)  #temporary Failure(8/10)
        profit=exit_alarm_enable(avg_price,close_price,position_side)
        margin = round(float(position['marginSize']),1)
        margin_rate = margin/total_div4*100
        breakeven = round(float(position['breakEvenPrice']),5)

        if live24data_condition !=0:
            total_absamount = (live24data['short_absamount']+live24data['long_absamount']+freeamount*float(leverage))
            half_absamount = total_absamount/2
            short_rate = (live24data['short_absamount']/half_absamount)*100
            long_rate = (live24data['long_absamount']/half_absamount)*100
            total_absamount_rate = (live24data['short_absamount']+live24data['long_absamount'])/total_absamount
            avg_break_gap= exit_alarm_enable(breakeven,avg_price,position_side)  #temporary Failure(8/10)
            profit=exit_alarm_enable(avg_price,close_price,position_side)
            close_price = float(marketApi.ticker(symbol,'USDT-FUTURES')['data'][0]['lastPr'])
            currency =CurrencyConverter('http://www.ecb.europa.eu/stats/eurofxref/eurofxref.zip')
            usd_krw=round(currency.convert(1,'USD','KRW'),1)

            chgUtc = abs(chgUtc)
            bet_sizex = round(freeamount*leverage/bet_sizex_div,1)

            avg_gap = live24data['long_avg_price']-live24data['short_avg_price']
            gap_middle = (live24data['long_avg_price']+live24data['short_avg_price'])/2
            gap_percent = avg_gap/(close_price/100)
            set_pull = (gap_percent+0.5)*(-1)
            free_cnt = round(freeamount*leverage*10000)
            ShortSafeMargin = 0.9
            LongSafeMargin = 1.1
            bet_size = bet_size_base*bet_sizex
            
            if position_side == 'short':
                if liquidationPrice*ShortSafeMargin<close_price:
                    if free >1:
                        short_entry() #가용 마진이 있으면 마진 투입부터
                    else:  # 마진이 없으면 일단, 손절후, 마진 투입
                        try:
                            print("entry>>liquidationPrice*0.9:{}<close:{}".format(liquidationPrice*0.9,close_price))
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                            trg_price = round(close_price*0.9998,1) # 2025-08-31
                            #result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            #myutil2.live24flag('highest_short_price',filename2,trg_price)
                            message = "[{}][liquidationPrice*0.98:{}<close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*0.98,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass

                if live24data['long_position_running']:
                    print("long_profit:{} > alpha*100:{}".format(long_profit, alpha*100))
                print("highest_short_price:{}*(1+{}:{}):{}<close_price:{}".format(live24data['highest_short_price'],live24data['short_gap_rate'],1+live24data['short_gap_rate'],live24data['highest_short_price']*(1+live24data['short_gap_rate']),close_price))
                condition1 = float(live24data['highest_short_price'])*(1+live24data['short_gap_rate'])<close_price
                if account == 'Sub10':
                    condition2 = reentry_filter
                    condition3 = entry_level < close_price
                else:   
                    condition2 = 0
                    condition3 = 0
                if condition1 or (condition2 and condition3):
                    try:  # cycle_entry_filter_hysteresis 추세일때만 진입하는 조건문 추가 예정
                        print("short entry/triggerPrice:{}".format(float(close_price)))
                        time.sleep(60)
                        myutil2.live24flag('highest_short_price',filename2,float(close_price))
                        orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_take_profit,1), timeInForceValue='normal')
                        profit=exit_alarm_enable(avg_price,close_price,position_side)
                    except:
                        message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                    minutem=0

            elif position_side == 'long':  # 포지션 롱일때
                if liquidationPrice*LongSafeMargin>close_price:
                    if free >1:
                        long_entry()
                    else:
                        try:
                            print("[freeless]entry>>liquidationPrice:{}*1.1:{}>close:{}".format(liquidationPrice,liquidationPrice*1.1,close_price))
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                            trg_price = round(close_price*1.0002,1) #2025-08-31
                            #result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            myutil2.live24flag('lowest_long_price',filename2,trg_price)
#                            message = "[freeless][{}][liquidationPrice*1.1:{}>close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*1.1,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            #tg_send(message)
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass
           
                print("lowest_long_price:{}*(1-{}:{}):{}>close_price:{}".format(live24data['lowest_long_price'],live24data['long_gap_rate'],1-live24data['long_gap_rate'],live24data['lowest_long_price']*(1-live24data['long_gap_rate']),close_price))
                if live24data['short_position_running']:
                    print("short_profit:{} > alpha*100:{}".format(short_profit, alpha*100))
                condition1 = float(live24data['lowest_long_price'])*(1-live24data['long_gap_rate'])>close_price
                if account == 'Sub10':
                    condition2 = reentry_filter
                    condition3 = entry_level > close_price
                else:   
                    condition2 = 0
                    condition3 = 0
                if condition1 or (condition2 and condition3):
                    if free > 1:  #cycle_entry_filter_hysteresis # cycle_entry_filter_hysteresis 추세일때만 진입하는 조건문 추가 예정
                        if account == 'Sub10': #short 포지션이 있고, short_profit이 alpha*100보다 클때만 진입, 다른 계정은 물려있어 해소 될때까지 진입 금지 
                            try:
                                orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='buy', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), timeInForceValue='normal',presetStopSurplusPrice=round(close_price*long_take_profit,1))
                                time.sleep(5)
                                myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                                profit=exit_alarm_enable(avg_price,close_price,position_side)
                            except:
                                message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                                minutem=0

            print("count:{}".format(cnt))
            time.sleep(10)

            if cnt%6 ==0:
                print("long_avg_price:{}-short_avg_price:{}={}/{}%".format(live24data['long_avg_price'],live24data['short_avg_price'],avg_gap,gap_percent))
                print("iquidationPrice:{}".format(liquidationPrice))
                gap_dynamic_cap = abs(T_market / T_control)

                if position_side == 'short':
                    short_take_profit0= 1-profit_base_rate
                    myutil2.live24flag('short_take_profit',filename2,short_take_profit0)
                    gap_effective = min(
                    gap_base_rate + gap_expend_rate*live24data['sell_orders_count'],
                    gap_dynamic_cap
                    )
                    gap_expend = gap_expend_rate*live24data['sell_orders_count']
                    myutil2.live24flag('short_gap_rate',filename2,gap_effective)
                    myutil2.live24flag('short_liquidationPrice',filename2,liquidationPrice)
                    myutil2.live24flag('short_absamount',filename2,absamount)
                    myutil2.live24flag('short_avg_price',filename2,avg_price)
                    myutil2.live24flag('short_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                    myutil2.live24flag('sell_orders_count',filename2,len(sell_orders))
                    # sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']))[0]
                    # myutil2.live24flag('lowest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    # sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    # myutil2.live24flag('highest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    if liquidationPrice*ShortSafeMargin>close_price:
                        short_exit()
                elif position_side == 'long':                     
                    long_take_profit0= 1+profit_base_rate
                    myutil2.live24flag('long_take_profit',filename2,long_take_profit0)
                    gap_effective = min(
                    gap_base_rate + gap_expend_rate*live24data['buy_orders_count'],
                    gap_dynamic_cap
                    )
                    myutil2.live24flag('long_gap_rate',filename2,gap_effective)
                    myutil2.live24flag('long_liquidationPrice',filename2,liquidationPrice)
                    myutil2.live24flag('long_absamount',filename2,absamount)
                    myutil2.live24flag('long_avg_price',filename2,avg_price)
                    myutil2.live24flag('long_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                    myutil2.live24flag('buy_orders_count',filename2,len(buy_orders))
                    # sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    # myutil2.live24flag('highest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
                    # sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']))[0]
                    # myutil2.live24flag('lowest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
                    print(message)
                    if liquidationPrice*LongSafeMargin<close_price:
                        long_exit()

            if position_side == 'short':
                if return_true_after_minutes(timeout,live24data['short_entry_time'])[0]:   # 30분 마다 1% 이익 세팅
                    myutil2.live24flag('short_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                    sorted_sell_orders_last = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                    first_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[0]
                    last_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[-1]

                    if avg_price*0.95 < live24data['long_liquidationPrice']:
                        sorted_sell_orders_last_price = round(avg_price*0.95,1) #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                        sorted_sell_orders_last_price_1 = avg_price*0.95 #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                    else:
                        sorted_sell_orders_last_price = live24data['long_liquidationPrice'] #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                        sorted_sell_orders_last_price_1 = live24data['long_liquidationPrice'] #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)

                    if avg_price*short_profit_line < close_price:
                        trigger_price0 = avg_price*short_profit_line
                    else:
                        trigger_price0 = close_price*short_profit_line_adjust  #0.998 <- 0.9998 2025-11-26

                    sell_orders_gap = abs(trigger_price0-sorted_sell_orders_last_price_1)  #0.998 -> 0.9998 2025-11-19
                    sell_orders_unitgap = sell_orders_gap/(live24data['sell_orders_count']+1)

                    for i in range(len(sell_orders)):
                        sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[i]
                        sorted_price_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[i]
                        trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i)),1))                               
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_sell_orders['size'],trigger_price,type(trigger_price).__name__,sorted_sell_orders['triggerPrice'],type(sorted_sell_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_sell_orders['size'])
                        time.sleep(1)
                    if pre_short_count != live24data['sell_orders_count']:
                        if profit > 0:
                            message = "[Short timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['short_absamount'],achievedProfits,round(trigger_price0,1),round(sell_orders_unitgap),round(sorted_sell_orders_last_price_1))
                            tg_send(message)
                            pre_short_count = live24data['sell_orders_count']
                else:
                    print("short 최저점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['short_entry_time'])[1]))
           
            # 포지션 조정효율 떨어지는 것은 리스크로 봐야한다. (손절 관련 로직 넣을껏)
            # APAE 인데. 실시간 피드백이 아니므로. 조정후 다시 체크해야한다. 예방 주사 개념  calc_APAE (로테이트 방식)
            # 손실률 / 투입 비율 컷 감안 -5% / 65%
            # 변동성에 따른 변동 갭 필요함 

            elif position_side == 'long':
                if return_true_after_minutes(timeout,live24data['long_entry_time'])[0]:
                    myutil2.live24flag('long_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                    sorted_buy_orders_last = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    
                    if avg_price*1.05 < live24data['short_liquidationPrice']:
                        sorted_buy_orders_last_price = live24data['short_liquidationPrice']  #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                        sorted_buy_orders_last_price_1 = live24data['short_liquidationPrice']  #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                    else:
                        sorted_buy_orders_last_price = round(avg_price*1.05,1) #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                        sorted_buy_orders_last_price_1 = avg_price*1.05 #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)

                    profit_line = long_profit_line
 
                    if avg_price*profit_line > close_price:
                        trigger_price0 = avg_price*profit_line
                    else:
                        trigger_price0 = close_price*long_profit_line_adjust  #0.998 <- 0.9998 2025-11-26

                    buy_orders_gap = abs(trigger_price0-sorted_buy_orders_last_price_1)  #1.002 2025-11-19
                    buy_orders_unitgap = buy_orders_gap/(live24data['buy_orders_count']+1)

                    for i in range(len(buy_orders)):
                        sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['size']),reverse=False)[i]
                        sorted_price_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[i]
                        trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i)),1))
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_buy_orders['size'],trigger_price,type(trigger_price).__name__,sorted_buy_orders['triggerPrice'],type(sorted_buy_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_buy_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_buy_orders['size'])
                        time.sleep(1)
                    if pre_long_count != live24data['buy_orders_count']:
                        if profit > 0:
                            message = "[Long timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['long_absamount'],achievedProfits,round(trigger_price0,1),round(buy_orders_unitgap),round(sorted_buy_orders_last_price_1))
                            tg_send(message)
                            pre_long_count = live24data['buy_orders_count']
                else:
                    print("long 최고점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['long_entry_time'])[1]))


