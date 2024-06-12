# start at 01/14 14:00 to 06/10 13:00
import csv
import statistics
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "./outputData/BTC20170101_20240609"
OUTPUT_PATH = "./outputData"
HEAD_INDEX = 5
TAIL_INDEX = 64997

def plot_data(data):
  plt_arr = []
  for i in range(len(data)):
    # print(data[i][2])
    plt_arr.append(data[i][3])

  # print(plt_arr)
  plt.figure(figsize=(20, 6))
  plt.plot(range(len(data)), plt_arr)
  plt.savefig(f"{OUTPUT_PATH}/BTC.png")

def str2float(data):
  for i in range(len(data)):
    for j in range(1, len(data[i])):
      data[i][j] = float(data[i][j])
  return data

def check_data(open, oyo, sigma, hl_in, co_in, sigma_in):
  if len(open) != len(oyo) or len(oyo) != len(sigma) or len(sigma) != len(hl_in) or len(hl_in) != len(co_in) or len(co_in) != len(sigma_in):
    print("[-] Data length error")
    print(f"open: {len(open)}")
    print(f"oyo: {len(oyo)}")
    print(f"sigma: {len(sigma)}")
    print(f"hl_in: {len(hl_in)}")
    print(f"co_in: {len(co_in)}")
    print(f"sigma_in: {len(sigma_in)}")
  else:
    print("[+] Data length check pass")

def calc_open(data):
  tmp = []
  for i in range(HEAD_INDEX, TAIL_INDEX):
    if (data[i][0][-2] == '1' and data[i][0][-1] == '4'):  
      tmp.append(data[i][3])
  
  return tmp

def calc_oyo(data):
  tmp = []
  for i in range(TAIL_INDEX):
    if (data[i][0][-2] == '1' and data[i][0][-1] == '4'):  
      if i == 5:
        tmp.append(0)
      else:
        tmp.append(data[i][3] - data[i-1][3])
  
  return tmp

def calc_sigma(data):
  tmp = []
  for i in range(HEAD_INDEX, TAIL_INDEX, 24):
    # print(i)
    if i == 5:
      tmp.append(0)
    else:
      daily_data = []
      for j in range(24):
        daily_data.append(data[i-j][3])
      tmp.append(statistics.stdev(daily_data))
  
  return tmp

def calc_hl_in(data):
  tmp = []
  for i in range(HEAD_INDEX, TAIL_INDEX, 24):
    daily_data_max = []
    daily_data_min = []
    for j in range(24):
      daily_data_max.append(data[i+j][1])
      daily_data_min.append(data[i+j][2])
    tmp.append(max(daily_data_max) - min(daily_data_min))
  
  return tmp

def calc_co_in(data):
  tmp = []
  for i in range(HEAD_INDEX, TAIL_INDEX, 24):
    open_price = data[i][3]
    close_price = data[i+23][6]
    tmp.append(close_price - open_price)

  return tmp

def calc_sigma_in(data):
  tmp = []
  for i in range(HEAD_INDEX, TAIL_INDEX, 24):
    daily_data = []
    for j in range(3):
      daily_data.append(data[i+j][6])
    tmp.append(statistics.stdev(daily_data))

  return tmp

header = []
data = []

with open(f'{FILE_PATH}.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i, row in enumerate(rows):
    if i == 0:
      header = row        
    else:
      data.append(row)      # Sort data by ascending order of time

open = []
oyo = []
sigma = []
hl_in = []
co_in = []
sigma_in = []
feature_header = ["Open", "OYO", "Sigma", "HL_in", "CO_in", "Sigma_in"]
data = str2float(data)

date = []
dt = datetime.datetime(2017, 1, 10, 14, 0, 0)
end = datetime.datetime(2024, 6, 10, 13, 0, 0)
step = datetime.timedelta(days=1)

while dt < end:
    date.append(dt.strftime('%Y-%m-%d'))
    dt += step
  
plot_data(data)                 # plot BTC price
open = calc_open(data)          # calculate open price
oyo = calc_oyo(data)            # calculate OYO - Open(今日交易時間開盤價) - YTD Open(昨日交易時間開盤價)
sigma = calc_sigma(data)        # calculate sigma yesterday 
hl_in = calc_hl_in(data)        # calculate high low in
co_in = calc_co_in(data)        # calculate close open in
sigma_in = calc_sigma_in(data)  # calculate sigma in
check_data(open, oyo, sigma, hl_in, co_in, sigma_in)

f = pd.DataFrame([])
f.insert(0, 'time', date)
f.insert(1, feature_header[0], open)
f.insert(2, feature_header[1], oyo)
f.insert(3, feature_header[2], sigma)
f.insert(4, feature_header[3], hl_in)
f.insert(5, feature_header[4], co_in)
f.insert(6, feature_header[5], sigma_in)
f.to_csv(f"{OUTPUT_PATH}/BTC_feature.csv", index_label=False, index=False)
print("[+] Feature extraction done")


