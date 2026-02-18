#-*-coding:utf-8-*-
from pickle import TRUE
import subprocess
import time
import datetime
from datetime import datetime, timezone, timedelta 
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

def calculate_atr(response, period=14):
    # 1. 응답 데이터에서 실제 캔들 리스트만 추출
    if isinstance(response, dict) and 'data' in response:
        candle_list = response['data']
    else:
        candle_list = response

    # 2. 데이터 개수 확인
    if not candle_list or len(candle_list) <= period:
        print(f"데이터 부족: 현재 {len(candle_list) if candle_list else 0}개")
        return None

    # 3. 비트겟 데이터 정렬 확인 (타임스탬프 기준 과거->현재 순으로 정렬)
    # 비트겟 V2는 보통 최신 데이터가 [0]번에 옵니다. 계산을 위해 뒤집어줍니다.
    if float(candle_list[0][0]) > float(candle_list[-1][0]):
        candle_list = candle_list[::-1]

    # 4. TR(True Range) 계산
    tr_list = []
    for i in range(1, len(candle_list)):
        high = float(candle_list[i][2])
        low = float(candle_list[i][3])
        prev_close = float(candle_list[i-1][4])
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_list.append(tr)

    # 5. ATR 계산 (Wilder's Smoothing)
    atr = sum(tr_list[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        
    return atr

def trend_strength(df, atr, span=50):
    close = df.iloc[:, 4].astype(float)

    if len(close) < span + 1:
        return 0.0   # 데이터 부족 → 추세 없음

    ema = close.ewm(span=span).mean()

    slope = ema.iloc[-1] - ema.iloc[-span]
    normalized_slope = slope / atr if atr > 0 else 0

    strength = np.tanh(abs(normalized_slope) * 3)
    return strength

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


if __name__ == "__main__":
    cnt=0
    cntm=0
    minute=0
    minutem=0
    pre_count=0
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
    timeout = 9 # 9분마다 1% 기타줄 세팅

    if coin == 'QQQUSDT':
        symbol = 'QQQUSDT'
        symbol2 = 'qqqusdt'
        entry_percent = 0.5 # 최초 진입점 및, 이후 진입 기준점 
        bet_sizex_div = 0.5 # 진입비율 계산용 계수 
        bet_size_base = 0.01 # 최소 진입 사이즈 
        gap_base_rate = 0.001 # 갭 계산용 계수
        gap_expend_rate = 0.0001 # 갭 확장 계산용 계수


    elif coin == 'BTCUSDT':
        symbol = 'BTCUSDT'
        symbol2 = 'btcusdt'
        entry_percent = 0.5
        bet_sizex_div = 0.005
        bet_size_base = 0.0001
        gap_base_rate = 0.002
        gap_expend_rate = 0.0001

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
        # 1. 데이터 가져오기 (충분한 데이터 확보를 위해 limit을 100 이상 권장)
        raw_data = marketApi.get_perp_candles("BTCUSDT", "1H", limit=100)
        current_atr = calculate_atr(raw_data, period=14)
        df = pd.DataFrame(raw_data, columns=cols)
        trend = trend_strength(df, current_atr, span=50)
        print(f"현재 BTCUSDT 1시간봉 ATR: {current_atr}")
        print("Trend Strength:", trend)
        close_price = float(marketApi.ticker(symbol,'USDT-FUTURES')['data'][0]['lastPr'])
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
        
        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')
        short_profit = live24data['short_profit'] #1.001 #live24data['long_take_profit']
        long_profit = live24data['long_profit'] #0.999 #live24data['short_take_profit']
        long_take_profit = live24data['long_take_profit'] #1.001 #live24data['long_take_profit']
        short_take_profit = live24data['short_take_profit'] #0.999 #live24data['short_take_profit']
        try:
            idx = get_pos_index(position,coin,position_side)
            if position_side == 'short':
                myutil2.live24flag('short_position_running',filename2,True)
            elif position_side == 'long':
                myutil2.live24flag('long_positon_running',filename2,True)
        except:
            print("Positon not found/Ready To entry State")
            # 모두 포지션 재개 조건 충족시 가장 작은 사이즈로 진입
            if position_side == 'short':
                myutil2.live24flag('short_position_running',filename2,False)
                if long_profit > entry_percent:
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_profit_line,1), timeInForceValue='normal')
                    message="[{}]1st Market Short Entry".format(account)
                    tg_send(message)
                    time.sleep(30)
            elif position_side == 'long':
                myutil2.live24flag('long_position_running',filename2,False)
                if short_profit > entry_percent:
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size_base,side='buy', tradeSide='open', marginMode='isolated', productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*long_profit_line,1), timeInForceValue='normal')
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
                            result = planApi.modify_tpsl_plan_v2(symbol=symbol2, marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            myutil2.live24flag('highest_short_price',filename2,trg_price)
                            message = "[{}][liquidationPrice*0.98:{}<close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*0.98,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass

                if live24data['long_position_running']:
                    print("long_profit:{} > entry_percent:{}".format(long_profit, entry_percent))
                print("highest_short_price*(1+{}:{}):{}<close_price:{}".format(live24data['short_gap_rate'],1+live24data['short_gap_rate'],live24data['highest_short_price']*(1+live24data['short_gap_rate']),close_price))
                if float(live24data['highest_short_price'])*(1+live24data['short_gap_rate'])<close_price:
                    if live24data['long_position_running'] and long_profit > entry_percent:
                        try:
                            orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_take_profit,1), timeInForceValue='normal')
                            time.sleep(5)
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                            print("short entry/triggerPrice:{}".format(sorted_sell_orders['triggerPrice']))
                            myutil2.live24flag('highest_short_price',filename2,float(close_price))
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
                            result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            myutil2.live24flag('lowest_long_price',filename2,trg_price)
                            message = "[freeless][{}][liquidationPrice*1.1:{}>close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*1.1,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            tg_send(message)
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass
           
                print("lowest_long_price*(1-{}:{}):{}>close_price:{}".format(live24data['long_gap_rate'],1-live24data['long_gap_rate'],live24data['lowest_long_price']*(1-live24data['long_gap_rate']),close_price))
                if live24data['short_position_running']:
                    print("short_profit:{} > entry_percent:{}".format(short_profit, entry_percent))
                if float(live24data['lowest_long_price'])*(1-live24data['long_gap_rate'])>close_price:
                    if free > 1:
                        if live24data['short_position_running'] and short_profit > entry_percent:
                            try:
                                orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='buy', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), timeInForceValue='normal',presetStopSurplusPrice=round(close_price*long_take_profit,1))
                                time.sleep(5)
                                result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                                buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                                sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']))[0]
                                myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                                profit=exit_alarm_enable(avg_price,close_price,position_side)
                            except:
                                message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                                minutem=0

            print("count:{}".format(cnt))
            time.sleep(10)

            if cnt%60 ==0:
                print("long_avg_price:{}-short_avg_price:{}={}/{}%".format(live24data['long_avg_price'],live24data['short_avg_price'],avg_gap,gap_percent))
                print("iquidationPrice:{}".format(liquidationPrice))
                if position_side == 'short':
                    gap_rate = gap_base_rate
                    short_take_profit0= 1-profit_base_rate
                    myutil2.live24flag('short_take_profit',filename2,short_take_profit0)
                    gap_expend = gap_expend_rate*live24data['sell_orders_count']
                    gap_rate = gap_rate+gap_expend
                    myutil2.live24flag('short_gap_rate',filename2,gap_rate)
                    myutil2.live24flag('short_liquidationPrice',filename2,liquidationPrice)
                    myutil2.live24flag('short_absamount',filename2,absamount)
                    myutil2.live24flag('short_avg_price',filename2,avg_price)
                    myutil2.live24flag('short_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == symbol]
                    myutil2.live24flag('sell_orders_count',filename2,len(sell_orders))
                    sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']))[0]
                    myutil2.live24flag('lowest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    myutil2.live24flag('highest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    if liquidationPrice*ShortSafeMargin>close_price:
                        short_exit()
                elif position_side == 'long':
                    gap_rate = gap_base_rate                        
                    long_take_profit0= 1+profit_base_rate
                    myutil2.live24flag('long_take_profit',filename2,long_take_profit0)
                    gap_expend = gap_expend_rate*live24data['buy_orders_count']
                    gap_rate = gap_rate+gap_expend
                    print("long_gap_rate:{}".format(gap_rate))
                    myutil2.live24flag('long_gap_rate',filename2,gap_rate)
                    myutil2.live24flag('long_liquidationPrice',filename2,liquidationPrice)
                    myutil2.live24flag('long_absamount',filename2,absamount)
                    myutil2.live24flag('long_avg_price',filename2,avg_price)
                    myutil2.live24flag('long_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == symbol]
                    myutil2.live24flag('buy_orders_count',filename2,len(buy_orders))
                    sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    myutil2.live24flag('highest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
                    sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']))[0]
                    myutil2.live24flag('lowest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
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
                        if i == 0:
                            trigger_price = sorted_sell_orders_last_price
                        else:
                            trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i-1)),1))                               
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_sell_orders['size'],trigger_price,type(trigger_price).__name__,sorted_sell_orders['triggerPrice'],type(sorted_sell_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_sell_orders['size'])
                        time.sleep(1)
                    if pre_count != live24data['sell_orders_count']:
                        if profit > 0:
                            message = "[Short timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['short_absamount'],achievedProfits,round(trigger_price0,1),round(sell_orders_unitgap),round(sorted_sell_orders_last_price_1))
                            tg_send(message)
                            pre_count = live24data['sell_orders_count']
                else:
                    print("short 최저점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['short_entry_time'])[1]))


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
                        if i ==0:
                            trigger_price = sorted_buy_orders_last_price
                        else:
                            trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i-1)),1))
                        print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_buy_orders['size'],trigger_price,type(trigger_price).__name__,sorted_buy_orders['triggerPrice'],type(sorted_buy_orders['triggerPrice']).__name__))
                        result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_buy_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_buy_orders['size'])
                        time.sleep(1)
                    if profit > 0:
                        message = "[Long timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['long_absamount'],achievedProfits,round(trigger_price0,1),round(buy_orders_unitgap),round(sorted_buy_orders_last_price_1))
                        tg_send(message)
                else:
                    print("long 최고점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['long_entry_time'])[1]))


