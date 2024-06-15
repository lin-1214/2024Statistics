import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

FILE_PATH = "../outputData/BTC20170101_20240609"
ORIGIN_PATH = "../outputData/BTC_label.csv"
PREDICT_PATH = "../predictModel/outputData/BTC_predict.csv"
IMG_PATH = "../img"
HEAD_INDEX = 5
TAIL_INDEX = 64997
HEAD_DAY = 1452
# TAIL_DAY = 1816
# HEAD_DAY = 2182
TAIL_DAY = 2547

def str2float(data):
  for i in range(len(data)):
    for j in range(1, len(data[i])):
      data[i][j] = float(data[i][j])
  return data

def calc_accumulate_profit(data, labels):
    accum_profit = 0
    tmp = []
    for i in range(len(labels)):
        weight = 2 - labels[i]
        flag = -1
        idx = HEAD_INDEX + 24 * i - 1
        delta = (data[idx][1] - data[idx][2]) / 3
        for j in range(24):
            if data[idx+j][6] > data[idx-1][1] + weight * delta:
                flag = j
                break
        
        if flag >= 0:
            accum_profit += data[idx+23][6] - data[idx+flag][6]
            tmp.append(accum_profit)
        else: 
            tmp.append(accum_profit)
  
    return tmp

data = []
origin_labels = []
predict_CNN_labels = []
predict_LSTM_labels = []
predict_TCN_labels = []
predict_Stack_labels = []

with open(ORIGIN_PATH, newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i, row in enumerate(rows):
    if i == 0:
      continue        
    else:
      data.append(row)      # Sort data by ascending order of time

for i in range(len(data)):
  origin_labels.append(data[i].index("1") - 1)

with open(PREDICT_PATH, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for i, row in enumerate(rows):
        if i == 0:
            continue        
        else:
            predict_LSTM_labels.append(int(row[1][0]))
            predict_CNN_labels.append(int(row[2][0]))
            predict_TCN_labels.append(int(row[3][0]))
            predict_Stack_labels.append(int(row[4][0]))

data = []
with open(f'{FILE_PATH}.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i, row in enumerate(rows):
    if i == 0:
      continue        
    else:
      data.append(row)      # Sort data by ascending order of time

data = str2float(data)
origin_profit = calc_accumulate_profit(data, origin_labels)
predict_CNN_profit = calc_accumulate_profit(data, predict_CNN_labels)
predict_LSTM_profit = calc_accumulate_profit(data, predict_LSTM_labels)
predict_TCN_profit = calc_accumulate_profit(data, predict_TCN_labels)
predict_Stack_profit = calc_accumulate_profit(data, predict_Stack_labels)

for i in range(HEAD_DAY, TAIL_DAY):
    origin_profit[i] -= origin_profit[HEAD_DAY-1]
    predict_CNN_profit[i] -= predict_CNN_profit[HEAD_DAY-1]
    predict_LSTM_profit[i] -= predict_LSTM_profit[HEAD_DAY-1]
    predict_TCN_profit[i] -= predict_TCN_profit[HEAD_DAY-1]
    predict_Stack_profit[i] -= predict_Stack_profit[HEAD_DAY-1]

plt.figsize = (20, 10)
plt.plot(origin_profit[HEAD_DAY: TAIL_DAY], label='origin')
plt.plot(predict_CNN_profit[HEAD_DAY: TAIL_DAY], label='CNN')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Origin vs CNN)")
plt.savefig(f"{IMG_PATH}/origin_CNN.png")

plt.clf()
plt.plot(origin_profit[HEAD_DAY: TAIL_DAY], label='origin')
plt.plot(predict_LSTM_profit[HEAD_DAY: TAIL_DAY], label='LSTM')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Origin vs LSTM)")
plt.savefig(f"{IMG_PATH}/origin_LSTM.png")

plt.clf()
plt.plot(origin_profit[HEAD_DAY: TAIL_DAY], label='origin')
plt.plot(predict_TCN_profit[HEAD_DAY: TAIL_DAY], label='TCN')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Origin vs TCN)")
plt.savefig(f"{IMG_PATH}/origin_TCN.png")

plt.clf()
plt.plot(origin_profit[HEAD_DAY: TAIL_DAY], label='origin')
plt.plot(predict_Stack_profit[HEAD_DAY: TAIL_DAY], label='Stack')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Origin vs Stack)")
plt.savefig(f"{IMG_PATH}/origin_Stack.png")

plt.clf()
plt.plot(origin_profit[HEAD_DAY: TAIL_DAY], label='origin')
plt.plot(predict_CNN_profit[HEAD_DAY: TAIL_DAY], label='CNN')
plt.plot(predict_LSTM_profit[HEAD_DAY: TAIL_DAY], label='LSTM')
plt.plot(predict_TCN_profit[HEAD_DAY: TAIL_DAY], label='TCN')
plt.plot(predict_Stack_profit[HEAD_DAY: TAIL_DAY], label='Stack')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Origin vs All)")
plt.savefig(f"{IMG_PATH}/origin_All.png")

plt.clf()
plt.plot(predict_CNN_profit[HEAD_DAY: TAIL_DAY], label='CNN')
plt.plot(predict_LSTM_profit[HEAD_DAY: TAIL_DAY], label='LSTM')
plt.plot(predict_TCN_profit[HEAD_DAY: TAIL_DAY], label='TCN')
plt.legend()
plt.title("2021~2023 Accumulate Profit (Compare 3 Models)")
plt.savefig(f"{IMG_PATH}/Mod3l.png")



