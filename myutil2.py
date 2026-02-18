#-*-coding:utf-8-*-
import time
import datetime
import ccxt
import telegram
import random
import hashlib
import hmac
from urllib.parse import urlparse
#import dbinsert
import base64, hashlib, hmac, json, requests, time
import sqlite3
import pandas as pd
from pandas import DataFrame
import threading

#conn=sqlite3.connect("/home/cha/Ftx/ftx.db", timeout=20)
#cur = conn.cursor()


def exchange(coin):
    if coin == 'ETH': #main account
        APIKEY = 'vXA6m6e31G6bk6dB5WJ1D3MXJN5IVCgvqAoTKddC'
        SECRET = 'G-uBs8sRhXQST-TqSZDheCUnUD1aAjQc9U7s6LoD'
    elif coin == 'BTC': #portfolio1 subaccount
        APIKEY = 'uuzhvZ594OYHCuiNTlTE7GDT-VrOwiFs-AsjvTPn'
        SECRET = '2t4m1Tc8bucRKDYlMYHAktBd2cNR-T-ZwzCYZPqy'
    ret = ccxt.ftx({
    'apiKey': APIKEY,
    'secret': SECRET,
    'enableRateLimit': True,})
    return ret


# telegram info 
my_token = '744382085:AAE-4OscklWp-YC7IhziQUH3djVun0IvIQY'
chat_id = '731080199'
bot = telegram.Bot(token = my_token)

def subaccount_transer(size, source, destination):
    config = {
        'apiKey': "vXA6m6e31G6bk6dB5WJ1D3MXJN5IVCgvqAoTKddC",
        'secret': "G-uBs8sRhXQST-TqSZDheCUnUD1aAjQc9U7s6LoD",
    }
    if source == 'Main':
        source = 'main'  
    if destination == 'Main':
        destination = 'main'
    config['headers'] = { 'FTX-SUBACCOUNT': None }
    client = ccxt.ftx(config)
    body2 = {
        "coin": "USD",
        "size": size,   
        "source": source,
        "destination": destination,  
    }
    print(body2)
    try:
        res=client.private_post_subaccounts_transfer(body2)
        if res['success']:
            message = "{} 에서 {}으로 {} USD를 보낸다 ".format(source,destination,size)
            print(message)
            bot.sendMessage(chat_id=chat_id, text=message)
            return True
    except:
        return False

def rsi(symbol, period: int = 14):
   lst=exchange_live.fetch_index_ohlcv(symbol,timeframe='5m',since=None,limit=500)
   columns=['timestamp','open','high','low','close','volume']
   df = DataFrame(lst, columns=columns)
   df["close"] = df["close"]
   delta = df["close"].diff()
   #ohlc["close"] = ohlc["close"]
   #delta = ohlc["close"].diff()
   gains, declines = delta.copy(), delta.copy()
   gains[gains < 0] = 0
   declines[declines > 0 ] = 0

   _gain = gains.ewm(com=(period -1), min_periods=period).mean()
   _loss = declines.abs().ewm(com=(period -1), min_periods=period).mean()

   RS = _gain / _loss
   return pd.Series(100 - (100/(1+RS)), name="RSI")


def DB_Vacuum(avgspan):
    avgspan = 60 * avgspan
    before = time.time() - avgspan 
    sql =  'delete from Price where Time < "{}";'.format(datetime.datetime.fromtimestamp(before))
    cur.execute(sql)
    conn.commit()
    time.sleep(5)
    sql =  "vacuum;"
    cur.execute(sql)
    conn.commit()

def DB_Vacuum_asset(avgspan):
    avgspan = 60 * avgspan
    before = time.time() - avgspan 
    sql =  'delete from Asset where Time < "{}";'.format(datetime.datetime.fromtimestamp(before))
    cur.execute(sql)
    conn.commit()
    time.sleep(5)
    sql =  "vacuum;"
    cur.execute(sql)
    conn.commit()

def rsiradarflag(n1,n2, value):
    with open('rsiradar.json', 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1][n2]=value
    #time.sleep(0.1)
    with open ('rsiradar.json', 'w') as w:
        json.dump(data, w)

def rsiflag(n1,n2, value):
    with open('rsi.json', 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1][n2]=value
    #time.sleep(0.1)
    with open ('rsi.json', 'w') as w:
        json.dump(data, w)

def flag(n1,n2, value):
    lock = threading.Lock()
    lock.acquire()
    with open('BTC-PERP.json', 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1][n2]=value
    #time.sleep(0.1)
    with open ('BTC-PERP.json', 'w') as w:
        json.dump(data, w)
    lock.release()

def flagwithfile(file,n1,n2, value):
    lock = threading.Lock()
    lock.acquire()
    with open(file, 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1][n2]=value
    #time.sleep(0.1)
    with open (file, 'w') as w:
        json.dump(data, w)
    lock.release()

def flagwithfile_onearg(file,n1, value):
    lock = threading.Lock()
    lock.acquire()
    with open(file, 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1]=value
    #time.sleep(0.1)
    with open (file, 'w') as w:
        json.dump(data, w)
    lock.release()

def updnflag(n1,n2, value):
    with open('updn.json', 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1][n2]=value
    #time.sleep(0.1)
    with open ('updn.json', 'w') as w:
        json.dump(data, w)

def get_sort(cl_list,mode,reverse):
#    with open('avglive24.json') as f:
#        live24data = json.load(f)
    dic = {}
    list = []
    for n in cl_list:
        filename = n+'.json'
        with open(filename) as f:
            live24data = json.load(f)
#        if live24data['isReadyExit']: 
        dic[n]=live24data[mode]
        print("[{}][{}%]".format(n,live24data[mode]))
#    print(dic)
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])        
    return list

def get_sort2(cl_list,mode,reverse):
    mdic = {}
    mlist = []
    pdic = {}
    plist = []
    for n in cl_list:
        filename = n+'.json'
        with open(filename) as f:
            live24data = json.load(f)
        if live24data[mode] > 0:
            pdic[n]=live24data[mode]
        elif live24data[mode] < 0:  # -2
            mdic[n]=live24data[mode]
#        print("[{}][{}]".format(n,live24data[mode]))
    print("mdic:{}".format(mdic))
    print("pdic:{}".format(pdic))
    if reverse == 'True':
        mres = sorted(mdic.items(),key=(lambda x:x[1]),reverse=True)
        pres = sorted(pdic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        mres = sorted(mdic.items(),key=(lambda x:x[1]),reverse=False)
        pres = sorted(pdic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(mdic)):
        mlist.append(mres[n][0])        
    for n in range(len(pdic)):
        plist.append(pres[n][0])        
    if len(mlist) != 0:
        return mlist,plist,mdic[mlist[len(mlist)-1]]
    else:
        return mlist,plist,-0.8

def get_sort_ratio(cl_list,mode,reverse,ratio):
    dic = {}
    list = []
    for n in cl_list:
        filename = n+'.json'
        with open(filename) as f:
            live24data = json.load(f)
        if live24data['ratio'] > ratio:
            if live24data[mode] < -1.5:
                dic[n]=live24data[mode]
#        print("[{}][{}]".format(n,live24data[mode]))
    print("ratio_dic:{}".format(dic))
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])        
    return list

def get_sort_over6(cl_list,mode,reverse,ratio):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    dic = {}
    list = []
    for n in cl_list:
        if live24data['0'][n][mode] > 6 and abs(live24data['0'][n]['ratio']) > ratio:
            dic[n]=live24data['0'][n][mode]
    print(dic)
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])        
    return list

def get_sort_ratio_old(cl_list,mode,reverse,ratio):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    dic = {}
    list = []
    for n in cl_list:
        if abs(live24data['0'][n]['ratio']) > ratio:
            dic[n]=live24data['0'][n][mode]
    print(dic)
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])
    print(list)        
    return list

def get_sort_under2(cl_list,mode,reverse,ratio):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    dic = {}
    list = []
    for n in cl_list:
        if live24data['0'][n][mode] < 2 and abs(live24data['0'][n]['ratio']) > ratio:
            dic[n]=live24data['0'][n][mode]
    print(dic)
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])        
    return list

def get_sort_underm2(cl_list,mode,reverse):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    dic = {}
    list = []
    for n in cl_list:
        if live24data['0'][n][mode] < -2:
            dic[n]=live24data['0'][n][mode]
    print(dic)
    if reverse == 'True':
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    for n in range(len(dic)):
        list.append(res[n][0])        
    return list

def get_sort_mode(cl_list,account,mode,reverse):
    with open('BTC-PERP.json') as f:
        live24data = json.load(f)
    #print(cl_list)
    dic = {}
    list = []
    for n in cl_list:
        if n != account:
            dic[n]=live24data[n][mode]
    #print(dic)
    if reverse:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    print(res)
    #for n in range(len(dic)):
    #    list.append(res[n][0])        
    return res #list

def get_sort_mode_list(cl_list,mode,reverse):
    with open('BTC-PERP.json') as f:
        live24data = json.load(f)
    #print(cl_list)
    dic = {}
    list = []
    for n in cl_list:
        dic[n]=live24data[n][mode]
    #print(dic)
    if reverse:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    print(res)
    for n in range(len(dic)):
        list.append(res[n][0])
    #print(list)        
    return list

def get_sort_mode_listwithfile(file,cl_list,mode,reverse):
    with open(file) as f:
        live24data = json.load(f)
    #print(cl_list)
    dic = {}
    list = []
    for n in cl_list:
        dic[n]=live24data[n][mode]
    #print(dic)
    if reverse:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=True)
    else:
        res = sorted(dic.items(),key=(lambda x:x[1]),reverse=False)
    #print(res)
    for n in range(len(dic)):
        list.append(res[n][0])
    #print(list)        
    return list

def get_total_ratio(list):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    sum = 0
    for n in list:
        sum = sum + abs(live24data['0'][n]['ratio'])
    return round(sum,1)

def get_total(list,mode):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    sum = 0
    for n in list:
        sum = sum + live24data['0'][n][mode]
    return round(sum,1)


def live24flag(n1,filename, value):
    lock = threading.Lock()
    lock.acquire()
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except:
            return
    data[n1]=value
    #time.sleep(0.1)
    with open (filename, 'w') as w:
        json.dump(data, w)
    lock.release()

def get_expeted_percent(a,b):
    if b>0:
        return round((a-b)/b*100,2)
    elif b<0 and a != 0:
        return round((b-a)/a*100,2)
    elif b==0:
        return 0

def get_total_usdValue(coin_list2):
    sum = 0
    for n in coin_list2:
        coindex = make_coin_list_all()[0].index(n)
        sum = sum + float(exchange_live.fetch_balance()['info']['result'][coindex]['usdValue'])
#        print (n,coindex,float(exchange_live.fetch_balance()['info']['result'][coindex]['usdValue']))
    return round(sum)+float(exchange_live.fetch_balance()['info']['result'][make_coin_list_all()[0].index('USD')]['usdValue'])

def get_total_expectedValue(coin_list2):
    with open('avglive24.json') as f:
        live24data = json.load(f)
    sum = 0
    for n in coin_list2:
        sum = sum + live24data['0'][n]['exit_expected_value']
#        print (n,live24data['0'][n]['exit_expected_value'])
    return round(sum)+float(exchange_live.fetch_balance()['info']['result'][make_coin_list_all()[0].index('USD')]['usdValue'])

def make_coin_list_all():
    balance = exchange_live.fetch_balance()['info']['result']
    coinusd_list = []
    coin_list = []
    for n in range(len(balance)) :
        coinusd_list.append(balance[n]['coin']+'/USD')
        coin_list.append(balance[n]['coin'])
    return coin_list,coinusd_list

#print(make_coin_list_all()[1])
 
#print(list(set(make_coin_list_all()[0]) -set(['USD'])))
#print(list(set(make_coin_list_all()[1])-set(['USD/USD'])))





