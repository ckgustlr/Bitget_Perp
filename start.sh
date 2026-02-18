#!/bin/bash
sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub7_QQQUSDT_Long --log-date-format="HH:mm" -- 'long' 'Sub7' 'QQQUSDT'
sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub7_QQQUSDT_Short --log-date-format="HH:mm" -- 'short' 'Sub7' 'QQQUSDT'


sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub8_BTCUSDT_Long --log-date-format="HH:mm" -- 'long' 'Sub8' 'BTCUSDT'
sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub8_BTCUSDT_Short --log-date-format="HH:mm" -- 'short' 'Sub8' 'BTCUSDT'


sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub9_BTCUSDT_Long --log-date-format="HH:mm" -- 'long' 'Sub9' 'BTCUSDT'
sudo pm2 start Bitget_Perp_ver0.1.py --interpreter python3 --name Sub9_BTCUSDT_Short --log-date-format="HH:mm" -- 'short' 'Sub9' 'BTCUSDT'



