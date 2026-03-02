#-*-coding:utf-8-*-
from pickle import TRUE
from enum import Enum
from collections import deque
import subprocess
import time
from copy import deepcopy
import datetime
#from datetime import datetime, timezone, timedelta 
import requests
import json
from pathlib import Path
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

class HedgeState(Enum):
    SAFE = 0
    WARNING = 1
    DANGER = 2

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

def classify_hedge_state(
    unrealizedPnL: float,
    marginSize: float,
    unrealized_loss_ratio: float,
    stress: float,  # 0 ~ 1
):
    """
    계좌 생존 판단 전용
    포지션 구조 신호 제거
    """

    if marginSize <= 0:
        return HedgeState.SAFE

    hedge_roe = abs(unrealizedPnL / marginSize)

    # -------------------------
    # Thresholds
    # -------------------------
    ROE_WARN = 0.30

    LOSS_WARN = 0.20
    LOSS_DANGER = 0.35

    STRESS_WARN = 0.60
    STRESS_DANGER = 0.80

    # -------------------------
    # DANGER (계좌 붕괴 위험)
    # -------------------------
    if (
        unrealized_loss_ratio > LOSS_DANGER
        or stress > STRESS_DANGER
    ):
        return HedgeState.DANGER

    # -------------------------
    # WARNING (공격 축소)
    # -------------------------
    if (
        hedge_roe > ROE_WARN
        or unrealized_loss_ratio > LOSS_WARN
        or stress > STRESS_WARN
    ):
        return HedgeState.WARNING

    # -------------------------
    # SAFE
    # -------------------------
    return HedgeState.SAFE

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

def return_true_after_minutes(delay_sec,timestamp):
    target_timestamp = timestamp + delay_sec  # n초 후의 타임스탬프 계산
    if time.time() > target_timestamp:
        ret=1
        diff= target_timestamp-time.time()
    else:
        ret=0
        diff= target_timestamp-time.time()
    return ret,diff

def apply_gap_velocity(base_factor, gap_v):
    """
    gap_v > 0 : 여유 공간이 늘고 있음 → 조금 더 욕심 가능
    gap_v < 0 : 공간이 줄고 있음 → 즉시 보수화
    """

    adjustment = -gap_v * 5  # 감도 계수 (튜닝 포인트)

    adjusted = base_factor + adjustment 

    return max(0.90, min(adjusted, 0.98))

def adjust_margin(position_side: str, action: str):
    """
    position_side: "long" or "short"
    action: "entry" or "exit"
    """

    # 1️⃣ liquidation multiplier 결정
    if position_side == "long":
        liq_factor = 1.1
        comparison_symbol = ">"
    else:
        liq_factor = 0.9
        comparison_symbol = "<"

    adjusted_liq = liquidationPrice * liq_factor

    print(f"{action}>> liquidationPrice*{liq_factor}:{adjusted_liq} {comparison_symbol} close:{close_price}")

    # 2️⃣ 시도할 비율 목록
    ratios = [0.08, 0.04, 0.02, 0.01]

    for r in ratios:
        try:
            cal_amount = round(margin * r)

            # exit이면 음수
            amount = cal_amount if action == "entry" else -cal_amount

            message = (
                f"{action}>>[{account}][{position_side}]"
                f"liq*{liq_factor}:{adjusted_liq} "
                f"{comparison_symbol} close:{close_price} "
                f"{'+' if action=='entry' else '-'}{int(r*100)}%:{cal_amount}"
            )

            result = accountApi.margin(
                symbol,
                marginCoin='USDT',
                productType='USDT-FUTURES',
                amount=amount,
                holdSide=position_side
            )

            print(message)
            print(result)
            return  # 성공하면 종료

        except Exception as e:
            continue

    # 3️⃣ 최후 fallback → 1 USDT
    try:
        amount = 1 if action == "entry" else -1

        message = (
            f"{action}>>[{account}][{position_side}]"
            f"liq*{liq_factor}:{adjusted_liq} "
            f"{comparison_symbol} close:{close_price} "
            f"{'+1USD' if action=='entry' else '-1USD'}"
        )

        result = accountApi.margin(
            symbol,
            marginCoin='USDT',
            productType='USDT-FUTURES',
            amount=amount,
            holdSide=position_side
        )

        print(message)
        print(result)

    except:
        pass

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

def compute_size_factor(state, apae_score, hedge_sensor):

    # 기본값
    size_factor = 1.0

    # 1️⃣ State 기반 1차 감속
    if state == HedgeState.WARNING:
        size_factor *= 0.7
    elif state == HedgeState.DANGER:
        size_factor *= 0.4

    # 2️⃣ APAE 기반 균형 감속 (0 ~ 0.4 영향)
    apae_factor = 1 - min(apae_score, 0.4)
    size_factor *= apae_factor

    # 3️⃣ Hedge Sensor 기반 밀집 감속 (완만하게)
    hedge_factor = 1 - 0.3 * hedge_sensor
    size_factor *= hedge_factor

    return max(0.2, size_factor)

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

def trend_strength_normalized(df, atr, span=21):
    """
    추세 강도 (0 ~ 1 정규화)
    EMA 기울기를 ATR 대비 정규화
    """
    if len(df) < span + 2 or atr == 0:
        return 0.0

    ema = df["close"].ewm(span=span, adjust=False).mean()

    slope = ema.iloc[-1] - ema.iloc[-2]

    raw_strength = abs(slope) / atr

    # 0.15 이상은 강한 추세로 간주
    normalized = min(raw_strength / 0.15, 1.0)

    return normalized
    
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

def calc_gap_ratio(current_price, liquidation_price, factor):
    anchor = liquidation_price * factor
    return abs(anchor - current_price) / current_price

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

def compute_T_control2(
    T_base,
    gap_sensor,
    hedge_sensor,
    account_stress,
    alpha,
    T_min,
    T_max
):
    stress = (
        0.5 * gap_sensor +
        0.3 * hedge_sensor +
        0.2 * account_stress
    )

    T = T_base * (1 - alpha * stress)
    return np.clip(T, T_min, T_max)

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

import numpy as np

def calc_exit_levels(
    coin: str,
    price: float,
    atr: float,
    remaining_count: int,
    side: str,                       # "long" | "short"
    trend_strength: float,           # 0 ~ 1 (현재는 곡률에만 사용)
    profit_ceiling: float,           # 🔥 외부에서 계산된 최종 목표 수익률
    base_profit: float = 0.01,        # 항상 유지되는 최소 1%
    vol_min: float = 0.5,
    vol_max: float = 2.0,
    profit_mul_min: float = 1.0,
    profit_mul_max: float = 5.0,      # 하드 캡 (최대 5%)
    curve_max: float = 2.5
):
    """
    가변 Exit Price 레벨 생성 함수
    - 최소 1% 이익은 항상 유지
    - 추세 폭발 시 외부에서 profit_ceiling 확장 가능 (최대 5%)
    - 가격 기준으로 실제 exit price 반환
    - 소수점 첫째 자리까지 반올림
    """

    if remaining_count <= 0:
        return []

    # 1️⃣ 상대 변동성 계산
    raw_vol_factor = atr / price
    vol_factor = np.clip(raw_vol_factor, vol_min, vol_max)

    # 2️⃣ profit ceiling 안전 클램프
    min_profit = base_profit * profit_mul_min
    max_profit = base_profit * profit_mul_max

    profit_ceiling = np.clip(
        profit_ceiling * vol_factor,
        min_profit,
        max_profit
    )

    # 3️⃣ 곡률 계수 (추세 강할수록 뒤쪽 확장)
    curve_power = 1.0 + trend_strength * (curve_max - 1.0)

    x = np.linspace(0, 1, remaining_count)
    curve = x ** curve_power

    profit_levels = base_profit + curve * (profit_ceiling - base_profit)

    # 4️⃣ 가격 레벨 변환
    if side.lower() == "short":
        exit_prices = price * (1 - profit_levels)
    elif side.lower() == "long":
        exit_prices = price * (1 + profit_levels)
    else:
        raise ValueError("side must be 'long' or 'short'")

    # 5️⃣ 소수점 첫째 자리 반올림
    if coin == "BTCUSDT":
        exit_prices = [round(p, 1) for p in exit_prices]
    elif coin == "QQQUSDT":
        exit_prices = [round(p, 2) for p in exit_prices]

    return exit_prices

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

def make_position_state(avg_price: float, size: float):
    return {
        "avg_price": avg_price,
        "size": size,
        "notional": avg_price * size
    }

def rotate_position(snapshot: dict, side: str, new_avg: float, new_size: float):
    #print(f"Rotating {side} position: new_avg={new_avg}, new_size={new_size}")
    tg_send(f"Rotating {side} position: new_avg={new_avg}, new_size={new_size}")
    assert side in ("short", "long")

    current = snapshot[side]["cur"]

    # cur → prev
    if current is not None:
        snapshot[side]["prev"] = deepcopy(current)

    # new cur
    snapshot[side]["cur"] = make_position_state(new_avg, new_size)

    snapshot["timestamp"] = int(time.time())

def calc_APAE(snapshot: dict):

    long_cur = snapshot["long"]["cur"]
    short_cur = snapshot["short"]["cur"]
    capital = snapshot["capital"]
    market = snapshot["market"]

    # 1. 질량 비대칭
    size_gap = long_cur["size"] - short_cur["size"]

    # 2. 평단 갭
    avg_gap = short_cur["avg_price"] - long_cur["avg_price"]

    # 3. 외부 유입 완충
    effective_equity = (
        capital["core"]
        + capital["recycled"]
        + capital["external_inflow"] * 0.3
    )

    free_ratio = capital["free"] / effective_equity

    # 4. 변동성 가중치
    volatility_weight = 1 + market["volatility"]

    # 5. APAE score
    apae_score = (
        abs(size_gap) * 0.4 +
        abs(avg_gap / market["current_price"]) * 0.3 +
        (1 - free_ratio) * 0.3
    ) * volatility_weight

    return {
        "size_gap": size_gap,
        "avg_gap": avg_gap,
        "free_ratio": free_ratio,
        "APAE_score": round(apae_score, 6)
    }

def load_snapshot(position_json):
    with open(position_json, "r") as f:
        return json.load(f)

def save_snapshot(snapshot, position_json):
    with open(position_json, "w") as f:
        json.dump(snapshot, f, indent=2)

def calc_usable_range(current_price, liquidation_price, side):
    """
    side: 'long' or 'short'
    """
    if side == 'long':
        # long은 청산가가 아래
        return (current_price - liquidation_price) / current_price
    else:
        # short는 청산가가 위
        return (liquidation_price - current_price) / current_price

def dynamic_factor(
    usable_range: float,
    phase: float,
    vol: float,
    min_factor=0.90,
    max_factor=0.98
):
    """
    usable_range : 청산까지 남은 상대 거리
    phase        : T_market (0~1 권장)
    vol          : ATR / price
    """

    # 시장 압력
    pressure = phase * vol

    # 핵심 로직
    raw = 1 - usable_range * (0.4 + 0.6 * pressure)

    # 안전 클램프
    return max(min_factor, min(raw, max_factor))

def calc_anchor_price(liquidation_price, factor, side):
    """
    liquidation_price : 거래소 기준 청산가
    factor            : dynamic_factor 결과
    side              : 'long' or 'short'
    """
    if side == 'long':
        # long은 청산가보다 위쪽으로
        return liquidation_price * (2 - factor)
    else:
        # short는 청산가보다 아래쪽으로
        return liquidation_price * factor

def calc_gap_ratio(current_price, anchor_price):
    return abs(anchor_price - current_price) / current_price

def dynamic_profit_factor_v2(
    vol,
    phase,
    base=1.0,
    usable_range=0.06,      # 6% 실질 공간
    vol_ref=0.01,
    vol_weight=0.6,
    phase_weight=0.8,
    curvature=1.5,         # 비선형 강도
    min_factor=0.94,       # 최대 6% 당김
    max_factor=1.02
):
    """
    vol   : ATR / price
    phase : 0~1 정규화된 trend_strength
    """

    # 1️⃣ 공격성 점수 (0~1 이상 가능)
    raw_aggression = (
        vol_weight * (vol / vol_ref) +
        phase_weight * phase
    )

    # 2️⃣ 비선형 증폭 (불태우기 구간에서 급격)
    aggression = np.tanh(raw_aggression ** curvature)

    # 3️⃣ 실제 가격 공간으로 변환
    factor = base - usable_range * aggression

    return float(np.clip(factor, min_factor, max_factor))

def calc_usable_range(liq_price, current_price, side="short"):
    if side == "short":
        raw = (liq_price - current_price) / liq_price
    else:  # long
        raw = (current_price - liq_price) / current_price

    # 음수 방지 + 하한선
    return max(0.0, raw)

def calc_expansion_score(
    trend_strength,
    ratio,              # T_control / abs(T_market)
    account_stress,
    hedge_sensor
):
    """
    0 ~ 1 사이 확장 점수
    """

    # 추세가 강할수록 ↑
    trend_component = trend_strength

    # 시장이 빠를수록 (ratio < 1) 추세 가능성 ↑
    speed_component = np.clip(1 - ratio, 0, 1)

    # 계좌 안정적일수록 ↑
    stability_component = np.clip(1 - account_stress, 0, 1)

    # 헤지가 약할수록 (진짜 방향 노출) ↑
    exposure_component = np.clip(1 - hedge_sensor, 0, 1)

    # 가중 평균
    score = (
        0.4 * trend_component +
        0.2 * speed_component +
        0.2 * stability_component +
        0.2 * exposure_component
    )

    return np.clip(score, 0, 1)

def compute_size_factor(state, apae_score, hedge_sensor):

    # 기본값
    size_factor = 1.0

    # 1️⃣ State 기반 1차 감속
    if state == HedgeState.WARNING:
        size_factor *= 0.7
    elif state == HedgeState.DANGER:
        size_factor *= 0.4

    # 2️⃣ APAE 기반 균형 감속 (0 ~ 0.4 영향)
    apae_factor = 1 - min(apae_score, 0.4)
    size_factor *= apae_factor

    # 3️⃣ Hedge Sensor 기반 밀집 감속 (완만하게)
    hedge_factor = 1 - 0.3 * hedge_sensor
    size_factor *= hedge_factor

    return max(0.2, size_factor)

def calc_profit_multiplier(
    base_profit,
    expansion_score,
    base_mul_min=1.0,
    base_mul_max=1.5,
    expansion_mul_max=5.0
):
    """
    기본 1~1.5
    확장 조건에서 최대 5까지
    """

    # 1️⃣ 기본 영역 (0 ~ 0.5)
    if expansion_score < 0.5:
        # 1.0 ~ 1.5 사이만 움직임
        base_zone = expansion_score / 0.5
        return base_mul_min + base_zone * (base_mul_max - base_mul_min)

    # 2️⃣ 확장 영역 (0.5 ~ 1.0)
    else:
        expansion_zone = (expansion_score - 0.5) / 0.5
        return base_mul_max + expansion_zone * (expansion_mul_max - base_mul_max)

def update_runtime(snapshot, close_price, vol, free):
    snapshot["capital"]["core"] = 500
    snapshot["capital"]["recycled"] = free
    snapshot["capital"]["external_inflow"] = 0 
    snapshot["capital"]["free"] = free
    snapshot["timestamp"] = time.time()
    snapshot["market"]["current_price"] = close_price
    snapshot["market"]["volatility"] = vol

def calc_exit_interval_fast(
    total_qty,
    remain_qty,
    base_exit_interval,
    gamma=1.4,
    min_ratio=0.25,
    max_interval=None
):
    if total_qty <= 0:
        return base_exit_interval

    ratio = max(0.0, min(1.0, remain_qty / total_qty))
    interval = base_exit_interval * (ratio ** gamma)

    min_interval = base_exit_interval * min_ratio

    if max_interval is not None:
        interval = min(interval, max_interval)

    return max(min_interval, interval)

def calc_account_stress(margin_ratio: float) -> float:
    if margin_ratio <= 0.5:
        return margin_ratio / 0.5 * 0.5          # 0 ~ 0.5
    elif margin_ratio <= 0.7:
        return 0.5 + (margin_ratio - 0.5) / 0.2 * 0.3   # 0.5 ~ 0.8
    else:
        return min(1.0, 0.8 + (margin_ratio - 0.7) / 0.15 * 0.2)

def calc_exit_interval_slow(
    total_qty,
    remain_qty,
    base_exit_interval,
    gamma=1.6,
    max_ratio=2.5,
    max_interval=None
):
    if total_qty <= 0:
        return base_exit_interval

    ratio = max(0.0, min(1.0, remain_qty / total_qty))

    # 질량이 줄수록 증가하는 시간
    interval_multiplier = 1 + ((1 - ratio) ** gamma) * (max_ratio - 1)

    interval = base_exit_interval * interval_multiplier

    if max_interval is not None:
        interval = min(interval, max_interval)

    return interval

def calc_account_stress(margin_ratio: float) -> float:
    if margin_ratio <= 0.5:
        return margin_ratio / 0.5 * 0.5          # 0 ~ 0.5
    elif margin_ratio <= 0.7:
        return 0.5 + (margin_ratio - 0.5) / 0.2 * 0.3   # 0.5 ~ 0.8
    else:
        return min(1.0, 0.8 + (margin_ratio - 0.7) / 0.15 * 0.2)

def log_hedge_state(
    position_side,
    account,
    hedge_state,
    apae_score,
    hedge_sensor,
    adjust_size_factor
):
    global _prev_hedge_state

    message = (
        f"position:{position_side}, "
        f"account:{account}, "
        f"hedge_state:{hedge_state}, "
        f"APAE:{apae_score:.4f}, "
        f"hedge_sensor:{hedge_sensor:.3f} "
        f"-> adjust_size_factor:{adjust_size_factor:.3f}"
    )

    # 항상 콘솔 출력
    print(message)

    # 상태 변경 시에만 텔레그램 전송
    if _prev_hedge_state is None:
        tg_send(f"[INIT] {message}")
    elif hedge_state != _prev_hedge_state:
        tg_send(f"[STATE CHANGE] {message}")
    _prev_hedge_state = hedge_state

if __name__ == "__main__":
    cnt=0
    cntm=0
    minute=0
    minutem=0
    pre_short_count=0
    pre_long_count=0
    pre_set_lev = 0
    short_max_count = 0
    long_max_count = 0
    max_count = 0
    long_flag = False
    short_flag = False
    productType = 'USDT-FUTURES'
    marginC = 'USDT'
    productT='umcbl'
    prev_close_short = None
    prev_close_long  = None
    _prev_hedge_state = None
    profit_base_rate = 0.01 # 1% tp 세팅
    short_profit_line = 0.99 # 기타줄 초기 레벨
    short_profit_line_adjust = 0.999 # 1% 넘었을때 close_price 기준으로 세팅갭 0.1%
    long_profit_line = 1.01 # 기타줄 초기 레벨
    long_profit_line_adjust = 1.001 # 1% 넘었을때 close_price 기준으로 세팅갭 0.1%


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
    position_json = coin+'_'+account+'_position.json'

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
        chgUtc = float(marketApi.ticker(symbol,'USDT-FUTURES')['data'][0]['changeUtc24h'])*100
        chgUtcWoAbs = chgUtc
        balances = accountApi.account(coin,'USDT-FUTURES', marginCoin=marginC)
        #print(balances)
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
        
        snapshot = load_snapshot(position_json)

        # 예: long 포지션 증량
        #rotate_position(snapshot, "long", new_avg=42100, new_size=0.9)

        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')
        #prev_close_short = live24data['prev_close_short']
        #prev_close_long = live24data['prev_close_long']
        long_take_profit = live24data['long_take_profit'] #1.001 #live24data['long_take_profit']
        short_take_profit = live24data['short_take_profit'] #0.999 #live24data['short_take_profit']
        short_profit = live24data['short_profit'] #1.001 #live24data['long_take_profit']
        long_profit = live24data['long_profit'] #0.999 #live24data

        if position_side == 'short':
            short_max_count = live24data['short_max_count']
            current_scale_index = live24data['sell_orders_count']
            ankor_price= live24data['short_ankor_price']
            Risk_Anchor= live24data['short_liquidationPrice']
            Market_Stress_Anchor = live24data['long_liquidationPrice']
            if current_scale_index > short_max_count:
                short_max_count = current_scale_index
                myutil2.live24flag('short_max_count',filename2,short_max_count) 
            max_count = short_max_count

        elif position_side == 'long':
            long_max_count = live24data['long_max_count']
            current_scale_index = live24data['buy_orders_count']
            ankor_price = live24data['long_ankor_price']
            Risk_Anchor= live24data['long_liquidationPrice']
            Market_Stress_Anchor = live24data['short_liquidationPrice']
            if current_scale_index > long_max_count:
                long_max_count = current_scale_index
                myutil2.live24flag('long_max_count',filename2,long_max_count)
            max_count = long_max_count
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
        T_market_normalized = trend_strength_normalized(df, atr, span=21)
        price = df["close"].iloc[-1]
        vol = atr/price
        print(f"현재 1시간봉 ATR/close_price: {vol}") 
        print("T_market Trend Strength:", T_market) 
        print(f"T_market_normalized: {T_market_normalized:.4f}")

        # snapshot["capital"]["core"] = 500
        # snapshot["capital"]["recycled"] = free
        # snapshot["capital"]["external_inflow"] = 0 
        # snapshot["capital"]["free"] = free
        # snapshot["timestamp"] = time.time()
        # snapshot["market"]["current_price"] = close_price
        # snapshot["market"]["volatility"] = vol

        alpha = atr/price

        Adjusted_Alpha_100 = live24data['Adjusted_Alpha_100']
        try:
            idx = get_pos_index(position,coin,position_side)
            if position_side == 'short':
                myutil2.live24flag('short_position_running',filename2,True)
            elif position_side == 'long':
                myutil2.live24flag('long_positon_running',filename2,True)   
        except:
            # 모두 포지션 재개 조건 충족시 가장 작은 사이즈로 진입
            if position_side == 'short':
                myutil2.live24flag('short_position_running',filename2,False)
                short_max_count = 0
                myutil2.live24flag('short_max_count',filename2,short_max_count)   
                condition1 = long_profit > Adjusted_Alpha_100
                condition2 = T_market > 0.05
                print("condition1:{} / condition2: {} / long_profit > Adjusted_Alpha_100: {} > {} / T_market:{} > 0.05".format(condition1, condition2, long_profit, Adjusted_Alpha_100, T_market))
                if condition1 and condition2: #can_enter_short and 0: #확실히 long 추세확인하고 진입 
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_profit_line,1), timeInForceValue='normal')
                    myutil2.live24flag('highest_short_price',filename2,float(close_price))
                    myutil2.live24flag('short_ankor_price',filename2,float(close_price))
                    message="[{}]1st Market Short Entry".format(account)
                    tg_send(message)
                    time.sleep(30)
            elif position_side == 'long':
                myutil2.live24flag('long_position_running',filename2,False)
                long_max_count = 0
                myutil2.live24flag('long_max_count',filename2,long_max_count)  
                condition1 = short_profit > Adjusted_Alpha_100
                condition2 = T_market < -0.05  
                print("condition1:{} / condition2: {} / short_profit > Adjusted_Alpha_100: {} > {} / T_market:{} < -0.05".format(condition1, condition2, short_profit, Adjusted_Alpha_100, T_market))
                if condition1 and condition2: #can_enter_long and 0: #확실히 short 추세확인하고 진입 
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='buy', tradeSide='open', marginMode='isolated', productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*long_profit_line,1), timeInForceValue='normal')
                    myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                    myutil2.live24flag('long_ankor_price',filename2,float(close_price))  
                    message="[{}]1st Market Long Entry".format(account)
                    tg_send(message)
                    time.sleep(30)
            time.sleep(10)
            continue

        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')['data'][idx]
        leverage = float(position["leverage"])
        marginRatio=round(float(position['marginRatio']),3)
        liquidationPrice=round(float(position['liquidationPrice']),1)
        breakeven = round(float(position['breakEvenPrice']),1)
        marginSize=round(float(position['marginSize']),1)
        unrealizedPnL=round(float(position['unrealizedPL']),1)
        #print(position)
        achievedProfits=round(float(position['achievedProfits']),1)
        avg_price = round(float(position['openPriceAvg']),3)
       
        usable = calc_usable_range(close_price, Risk_Anchor, position_side)

        factor = dynamic_factor(
            usable_range=usable,
            phase=T_market,
            vol=vol
        )

        anchor = calc_anchor_price(
            liquidation_price=Risk_Anchor,
            factor=factor,
            side=position_side
        )

        gap_hist = deque(maxlen=5)

        # 1. 기본 환경 계산
        usable = calc_usable_range(close_price, Risk_Anchor, position_side)
        base_factor = dynamic_factor(usable, T_market, vol)

        # 2. anchor 계산
        anchor = calc_anchor_price(Risk_Anchor, base_factor, position_side)

        # 3. gap 계산
        gap_ratio = calc_gap_ratio(close_price, anchor)
        gap_hist.append(gap_ratio)

        print(f"gap_hist: {list(gap_hist)}")
        #gap_hist = [0.012, 0.014]
        # 4. gap velocity 계산
        gap_v = gap_velocity(gap_hist)

        # 5. factor 미세 조정
        final_factor = apply_gap_velocity(base_factor, gap_v)

        Profit_Expansion_Anchor = Market_Stress_Anchor*final_factor
    
        gap_sensor = np.clip(gap_v / T_PARAMS['gap_scale'] , 0, 1)
        hedge_roe = unrealizedPnL / marginSize
        hedge_roe_abs = abs(unrealizedPnL) / marginSize
        hedge_market_move = hedge_roe / leverage
        HEDGE_MOVE_MAX = 0.02   # 2%
        hedge_sensor = np.clip(abs(hedge_market_move) / HEDGE_MOVE_MAX, 0, 1)
        SAFE_MARGIN_RATIO = 0.1
        account_stress = calc_account_stress(marginRatio)
        alpha_base = 0.006   # 기본은 지금보다 약간 낮게
        k1 = 1.2             # 계좌 압박 가중치
        k2 = 0.8             # 헤지 약화 가중치
        alpha_dynamic = alpha_base * (
            1 + k1 * account_stress
        ) * (
            1 + k2 * (1 - hedge_sensor)
        )
        alpha_dynamic = np.clip(alpha_dynamic, 0.004, 0.02)  # 0.2% ~ 1% 사이
        #print( T_PARAMS['T_base'],T_PARAMS['T_min'], T_PARAMS['T_max'] )
        print(f"gap_sensor: {gap_sensor:.4f}, hedge_sensor: {hedge_sensor:.4f}, account_stress: {account_stress:.4f}")
        print(f"alpha_base: {alpha_base:.4f}, alpha_dynamic: {alpha_dynamic:.4f}")
        T_control = compute_T_control2(T_PARAMS['T_base'], gap_sensor, hedge_sensor, account_stress, alpha_dynamic, T_PARAMS['T_min'], T_PARAMS['T_max'])
        print("T_control (bars):", T_control)
        base_time = 900
        ratio = (T_control / abs(T_market))
        ratio = max(0.7, min(ratio, 2.0))
        exit_interval = base_time * ratio

        hedge_state = classify_hedge_state(unrealizedPnL,marginSize,hedge_roe_abs,account_stress)
        print(f"Hedge State: {hedge_state}, unrealizedPnL: {unrealizedPnL}, marginSize: {marginSize}, hedge_roe_abs: {hedge_roe_abs}, account_stress: {account_stress}")

        if hedge_state == HedgeState.SAFE:
            allow_new_entry = True
            allow_scale_in = False
            tighten_exit = False
            force_deleveraging = False

        if hedge_state == HedgeState.WARNING:
            allow_new_entry = True
            allow_scale_in = True
            tighten_exit = True
            force_deleveraging = False

        if hedge_state == HedgeState.DANGER:
            allow_new_entry = False
            allow_scale_in = False
            tighten_exit = True
            force_deleveraging = True
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
        myutil2.live24flag('Adjusted_Alpha_100',filename2,float(alpha*100))        
        print(f"Gap (price): {price * alpha}")


        if account == 'Sub10' or account == 'Sub7':
            print("position_side: {}, close_price: {}, avg_price: {}, ankor_price: {}, gap_base_rate: {}, gap_expend_rate: {}, current_scale_index: {}, T_market: {}".format(position_side, close_price, avg_price, ankor_price, gap_base_rate, gap_expend_rate, current_scale_index, T_market))
            reentry_filter = can_rebuild_position(position_side,close_price,avg_price,ankor_price,float(gap_base_rate),float(gap_expend_rate),current_scale_index,T_market,0.02)
            entry_level = get_next_entry_level(avg_price,position_side,gap_base_rate,gap_expend_rate,current_scale_index + 1)
            #entry_level = get_entry_level(avg_price, position_side, float(gap_base_rate), float(gap_expend_rate),current_scale_index+1)
            if position_side == 'short':
                print("Short Reentry Filter: {}, Short Entry Level: {}".format(reentry_filter, entry_level))
            elif position_side == 'long':
                print("Long Reentry Filter: {}, Long Entry Level: {}".format(reentry_filter, entry_level))
        
        delay_sec = exit_interval # 9분마다 1% 기타줄 세팅 무식해..ㅋ 
        if hedge_state == HedgeState.SAFE:
            adjust_delay_slow = calc_exit_interval_slow(total_qty=max_count, remain_qty=current_scale_index, base_exit_interval=exit_interval)
            adjust_delay_sec = adjust_delay_slow
            print("delay_sec: {}->{}/{}->adjust_delay_slow:{}".format(delay_sec, current_scale_index, max_count, adjust_delay_slow))
        else:
            adjust_delay_fast = calc_exit_interval_fast(total_qty=max_count, remain_qty=current_scale_index, base_exit_interval=exit_interval)
            adjust_delay_sec = adjust_delay_fast
            print("delay_sec: {}->{}/{}->adjust_delay_fast:{}".format(delay_sec, current_scale_index, max_count, adjust_delay_fast))

        print("Calculating exit levels with price: {}, atr: {}, remaining_count: {},max_count: {}, positionside: {}, trend_strength: {}".format(close_price, atr, current_scale_index, max_count, position_side, T_market))
        expansion_score = calc_expansion_score(
            T_market,
            ratio,
            account_stress,
            hedge_sensor
        )

        profit_multiplier = calc_profit_multiplier(
            0.01,  # base_profit (1% 목표)
            expansion_score
        )

        profit_ceiling = 0.01 * profit_multiplier
        print(f"Expansion Score: {expansion_score:.4f}, Profit Multiplier: {profit_multiplier:.4f}, Profit Ceiling: {profit_ceiling:.4f}")

        exit_levels =calc_exit_levels(
            coin,
            avg_price,
            atr,
            current_scale_index,
            position_side,                       # "long" | "short"
            T_market_normalized,           # 0 ~ 1 (현재는 곡률에만 사용)
            profit_ceiling   # 🔥 외부에서 계산된 최종 목표 수익률
        )

        print(exit_levels)

 #       print("close_price:{}/avg_price:{}".format(close_price,avg_price))
        absamount = float(position['available'])

        short_gap = abs(close_price-live24data['short_avg_price'])
        long_gap = abs(close_price-live24data['long_avg_price'])

        absamount_gap = abs(live24data['short_absamount']-live24data['long_absamount'])
        free_lev = float(freeamount) * float(leverage)
        short_lev = float(live24data['short_absamount']) * float(leverage)
        long_lev = float(live24data['long_absamount']) * float(leverage)
        #print("freeamount:{}/short:{}/long:{}/leverage:{}".format(freeamount,short_lev,long_lev,leverage))
        try:
            set_lev = float(absamount_gap) * float(leverage) /free_lev
        except:
            set_lev = 10
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

            APAE = calc_APAE(snapshot)
            print(APAE)
            adjust_size_factor = compute_size_factor(hedge_state, APAE['APAE_score'], hedge_sensor)
            log_hedge_state(
                position_side,
                account,
                hedge_state,
                APAE['APAE_score'],
                hedge_sensor,
                adjust_size_factor
            )

            if position_side == 'short':
                if live24data['long_position_running']:
                    print("long_profit:{} > alpha*100:{}".format(long_profit, alpha*100))
                print("highest_short_price:{}*(1+{}:{}):{}<close_price:{}".format(live24data['highest_short_price'],live24data['short_gap_rate'],1+live24data['short_gap_rate'],live24data['highest_short_price']*(1+live24data['short_gap_rate']),close_price))
                condition1 = float(live24data['highest_short_price'])*(1+live24data['short_gap_rate'])<close_price
                if account == 'Sub10' or account == 'Sub7':
                    condition2 = reentry_filter or hedge_state == HedgeState.SAFE
                    condition3_short = (
                        prev_close_short is not None and
                        prev_close_short <= entry_level and
                        close_price > entry_level
                    )
                else:   
                    condition2 = 0
                    condition3 = 0
                    condition3_short = 0
                print("condition1: {}, condition2: {}, condition3_short: {}".format(condition1, condition2, condition3_short))
                if hedge_state != HedgeState.DANGER:
                    if condition1 or (condition2 and condition3_short):
                        try:  
                            print("short entry/triggerPrice:{}".format(float(close_price)))
                            myutil2.live24flag('highest_short_price',filename2,float(close_price))
                            if hedge_state == HedgeState.WARNING:
                                if coin == "BTCUSDT":
                                    bet_size = round(bet_size*adjust_size_factor,4)
                                elif coin == "QQQUSDT":
                                    bet_size = round(bet_size*adjust_size_factor,2)
                            orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_take_profit,1), timeInForceValue='normal')
                            if condition3_short:
                                message="[ReUseVol][{}/{}][entry:{}][count:{}][state:{}]".format(account,coin,lose_price,current,current_scale_index,hedge_state)
                                tg_send(message)     
                            time.sleep(5)
                            position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')['data'][idx]
                            avg_price = round(float(position['openPriceAvg']),3)
                            rotate_position(snapshot, position_side, new_avg=avg_price, new_size=bet_size)
                            update_runtime(snapshot, close_price, vol, free)
                            profit=exit_alarm_enable(avg_price,close_price,position_side)
                        except:
                            message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                        minutem=0

            elif position_side == 'long':  # 포지션 롱일때          
                print("lowest_long_price:{}*(1-{}:{}):{}>close_price:{}".format(live24data['lowest_long_price'],live24data['long_gap_rate'],1-live24data['long_gap_rate'],live24data['lowest_long_price']*(1-live24data['long_gap_rate']),close_price))
                if live24data['short_position_running']:
                    print("short_profit:{} > alpha*100:{}".format(short_profit, alpha*100))
                condition1 = float(live24data['lowest_long_price'])*(1-live24data['long_gap_rate'])>close_price
                if account == 'Sub10' or account == 'Sub7':
                    condition2 = reentry_filter or hedge_state == HedgeState.SAFE# 유동성 재활용 필터 off 적용은 SAFE 때만 적용해보고, 추이를 모니터링  
                    condition3_long = (
                        prev_close_long is not None and
                        prev_close_long >= entry_level and
                        close_price < entry_level
                    )
                else:   
                    condition2 = 0
                    condition3 = 0
                    condition3_long = 0
                print("condition1: {}, condition2: {}, condition3_long: {}".format(condition1, condition2, condition3_long))    
                if hedge_state != HedgeState.DANGER:
                    if condition1 or (condition2 and condition3_long):
                        if free > 1:  #cycle_entry_filter_hysteresis # cycle_entry_filter_hysteresis 추세일때만 진입하는 조건문 추가 예정
                            if account == 'Sub10' or account == 'Sub7': #short 포지션이 있고, short_profit이 alpha*100보다 클때만 진입, 다른 계정은 물려있어 해소 될때까지 진입 금지 
                                try:
                                    if hedge_state == HedgeState.WARNING:
                                        if coin == "BTCUSDT":
                                            bet_size = round(bet_size*adjust_size_factor,4)
                                        elif coin == "QQQUSDT":
                                            bet_size = round(bet_size*adjust_size_factor,2)
                                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='buy', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), timeInForceValue='normal',presetStopSurplusPrice=round(close_price*long_take_profit,1))
                                    if condition3_long:
                                        message="[ReUseVol][{}/{}][entry:{}][count:{}][state:{}]".format(account,coin,lose_price,current,current_scale_index,hedge_state)
                                        tg_send(message)                                       
                                    time.sleep(5)
                                    position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')['data'][idx]
                                    avg_price = round(float(position['openPriceAvg']),3)
                                    rotate_position(snapshot, position_side, new_avg=avg_price, new_size=bet_size)
                                    update_runtime(snapshot, close_price, vol, free)
                                    myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                                    profit=exit_alarm_enable(avg_price,close_price,position_side)
                                except:
                                    message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                                    minutem=0

            save_snapshot(snapshot, position_json)

            print("count:{}".format(cnt))
            prev_close_short = close_price
            prev_close_long  = close_price
            myutil2.live24flag('prev_close_short',filename2,close_price)
            myutil2.live24flag('prev_close_long',filename2,close_price)            
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
                    if liquidationPrice*ShortSafeMargin>close_price:
                        adjust_margin("short", "exit")
                    else:
                        adjust_margin("short", "entry")
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
                    print(message)
                    if liquidationPrice*LongSafeMargin<close_price:
                        adjust_margin("long", "exit")
                    else:
                        adjust_margin("long", "entry")

            if position_side == 'short':
                if return_true_after_minutes(adjust_delay_sec,live24data['short_entry_time'])[0]:   # 30분 마다 1% 이익 세팅
                    myutil2.live24flag('short_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                    sorted_sell_orders_last = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                    first_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[0]
                    last_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[-1]

                    sorted_sell_orders_last_price = round(Profit_Expansion_Anchor,1)
                    sorted_sell_orders_last_price_1 = sorted_sell_orders_last_price

                    if avg_price*short_profit_line < close_price:
                        trigger_price0 = avg_price*short_profit_line
                    else:
                        trigger_price0 = close_price*short_profit_line_adjust  #0.998 <- 0.9998 2025-11-26

                    sell_orders_gap = abs(trigger_price0-sorted_sell_orders_last_price_1)  #0.998 -> 0.9998 2025-11-19
                    sell_orders_unitgap = sell_orders_gap/(live24data['sell_orders_count']+1)

                    for i in range(len(sell_orders)):
                        sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[i]
                        sorted_price_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[i]
                        if account == 'Sub7' or account == 'Sub10': #long 포지션이 있고, long_profit이 alpha*100보다 클때만 진입, 다른 계정은 물려있어 해소 될때까지 진입 금지
                            if HedgeState.SAFE != hedge_state and 0:  #SAFE 상태가 아닐때는 계정별로 조정된 이익실현 가격으로 TP 조정, SAFE 상태일때는 일반적인 조정 방식으로 TP 조정
                                trigger_price = exit_levels[i]
                            else:
                                trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i)),1))
                        else:
                            trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i)),1))                               
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_sell_orders['size'],trigger_price,type(trigger_price).__name__,sorted_sell_orders['triggerPrice'],type(sorted_sell_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_sell_orders['size'])
                        time.sleep(1)
                    if pre_short_count != live24data['sell_orders_count']:
                        if profit > 0:
                            message = "[{}][Short adjust_delay_sec:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(account,adjust_delay_sec,i,live24data['short_absamount'],achievedProfits,round(trigger_price0,1),round(sell_orders_unitgap),round(sorted_sell_orders_last_price_1))
                            tg_send(message)
                            pre_short_count = live24data['sell_orders_count']
                else:
                    print("short 최저점 조정 남은 시간:{}".format(return_true_after_minutes(adjust_delay_sec,live24data['short_entry_time'])[1]))
           

            # APAE 인데. 실시간 피드백이 아니므로. 조정후 다시 체크해야한다. 예방 주사 개념  calc_APAE (로테이트 방식) -> 포지션 투입하여 JSON 관찰필요 
            # -> 포지션 조정효율 떨어지는 것은 리스크로 봐야한다. (손절 관련 로직 넣을껏) 여기에 
            # 손실률 / 투입 비율 컷 감안 -5% / 65%
            # 변동성에 따른 변동 갭 필요함  -> 적용했으나 팩터 체크중   

            elif position_side == 'long':
                if return_true_after_minutes(adjust_delay_sec,live24data['long_entry_time'])[0]:
                    myutil2.live24flag('long_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                    sorted_buy_orders_last = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    
                    sorted_buy_orders_last_price = round(Profit_Expansion_Anchor,1)  #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                    sorted_buy_orders_last_price_1 = sorted_buy_orders_last_price #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)

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
                        if account == 'Sub7' or account == 'Sub10': 
                            if HedgeState.SAFE != hedge_state and 0:  #SAFE 상태가 아닐때는 계정별로 조정된 이익실현 가격으로 TP 조정, SAFE 상태일때는 일반적인 조정 방식으로 TP 조정
                                trigger_price = exit_levels[i]
                            else:
                                trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i)),1))
                        else:
                            trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i)),1))
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_buy_orders['size'],trigger_price,type(trigger_price).__name__,sorted_buy_orders['triggerPrice'],type(sorted_buy_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_buy_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_buy_orders['size'])
                        time.sleep(1)
                    if pre_long_count != live24data['buy_orders_count']:
                        if profit > 0:
                            message = "[{}][Long adjust_delay_sec:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(account,adjust_delay_sec,i,live24data['long_absamount'],achievedProfits,round(trigger_price0,1),round(buy_orders_unitgap),round(sorted_buy_orders_last_price_1))
                            tg_send(message)
                            pre_long_count = live24data['buy_orders_count']
                else:
                    print("long 최고점 조정 남은 시간:{}".format(return_true_after_minutes(adjust_delay_sec,live24data['long_entry_time'])[1]))

            # if hedge_state == HedgeState.DANGER:
            #     intensity = calc_cut_intensity(APAE, hedge_sensor)
            #     cut_ratio = calc_cut_ratio(intensity)
            # 모니터링 포인트
            # DANGER 진입 빈도

            # DANGER 체류 시간

            # DANGER 중 신규 진입 여부 (0이면 성공)

            # WARNING ↔ SAFE 복귀 속도

            # SAFE에서의 평단 개선 속도

            # 👉 이 다섯 개만 봐도
            # “투입 제어만으로 위험이 얼마나 흡수되는지”가 나온다.

            if force_deleveraging and 0:
                if position_side == 'short':
                    try:
                        print("[force_deleveraging]>liquidationPrice*0.9:{}<close:{}".format(liquidationPrice*0.9,close_price))
                        result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                        sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                        sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                        trg_price = round(close_price*0.9998,1) # 2025-08-31
                        #result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                        myutil2.live24flag('highest_short_price',filename2,trg_price)
                        message = "[force_deleveraging][{}][liquidationPrice*0.98:{}<close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*0.98,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                        time.sleep(8)
                    except:
                        pass

                if position_side == 'long':
                    try:
                        print("[force_deleveraging]>liquidationPrice:{}*1.1:{}>close:{}".format(liquidationPrice,liquidationPrice*1.1,close_price))
                        result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                        sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                        sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                        trg_price = round(close_price*1.0002,1) #2025-08-31
                        #result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                        myutil2.live24flag('lowest_long_price',filename2,trg_price)
                        message = "[force_deleveraging][{}][liquidationPrice*1.1:{}>close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*1.1,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                        #tg_send(message)
                        time.sleep(8)
                    except:
                        pass



