#!/usr/bin/python

from ..client import Client
from ..consts import *


class PositionApi(Client):
    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, first=False):
        Client.__init__(self, api_key, api_secret_key, passphrase, use_server_time, first)

    '''
    Obtain the user's single position information
    :return:
    '''
    def single_position(self, symbol, marginCoin, holdSide):
        params = {}
        if symbol:
            params["symbol"] = symbol
            params["marginCoin"] = marginCoin
            params["holdSide"] = holdSide
            return self._request_with_params(GET, MIX_POSITION_V1_URL + '/singlePosition', params)
        else:
            return "pls check args"

    '''
    Obtain all position information of the user
    productType: Umcbl (USDT professional contract) dmcbl (mixed contract) sumcbl (USDT professional contract simulation disk) sdmcbl (mixed contract simulation disk)
    :return:
    '''
    def all_position(self, productType, marginCoin):
        params = {}
        if productType:
            params["productType"] = productType
            params["marginCoin"] = marginCoin
            #return self._request_with_params(GET, MIX_POSITION_V1_URL + '/allPosition', params)
            return self._request_with_params(GET, MIX_POSITION_V2_URL + '/all-position', params)
        else:
            return "pls check args"
        
    def history_position(self, symbol, productType, start_time, end_time):
        params = {}
        if symbol:
            params["symbol"] = symbol
            params["productType"] = productType
            params["startTime"] = start_time
            params["endTime"] = end_time
        return self._request_with_params(GET, MIX_POSITION_V1_URL + '/history-position', params)

    #/api/v2/mix/position/history-position

    def history_position_V2(self, symbol, productType, start_time, end_time,limit):
        params = {}
        if symbol:
            params["symbol"] = symbol
            params["productType"] = productType
            params["startTime"] = start_time
            params["endTime"] = end_time
            params["limit"] = limit
        return self._request_with_params(GET, '/api/v2/mix/position/history-position', params)