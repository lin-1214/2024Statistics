import csv
import statistics
import datetime
import pandas as pd

FILE_PATH = "./outputData/BTC20170101_20240609"
OUTPUT_PATH = "./outputData"
HEAD_INDEX = 5
TAIL_INDEX = 64997

def str2float(data):
  for i in range(len(data)):
    for j in range(1, len(data[i])):
      data[i][j] = float(data[i][j])
  return data

def calc_pl(data, type):
    tmp = []
    weight = 3 - type   # weight * delta
    for i in range(HEAD_INDEX, TAIL_INDEX, 24):
        flag = -1
        delta = (data[i-1][1] - data[i-1][2]) / 3
        for j in range(24):
            if data[i+j][6] > data[i-1][1] + weight * delta:
                flag = j
                break
        
        if flag >= 0:
            tmp.append(data[i+23][6] - data[i+flag][6])
        else: 
            tmp.append(0)
        
    return tmp

def calc_sharpe(data):
    tmp = []
    for i in range(0, len(data)):
        if i < 7 or statistics.stdev(data[i-7:i]) == 0:
            tmp.append(0)
            continue
        sharpe = statistics.mean(data[i-7:i]) / statistics.stdev(data[i-7:i])
        tmp.append(sharpe)
    return tmp

def one_hot_encoding(data1, data2, data3, data4, data5):
    tmp = []
    n = len(data1)
    for i in range(n):
        row = []
        row.append(data1[i])
        row.append(data2[i])
        row.append(data3[i])
        row.append(data4[i])
        row.append(data5[i])
        max_index = row.index(max(row))
        one_hot = [0, 0, 0, 0, 0]
        one_hot[max_index] = 1
        tmp.append(one_hot)
    return tmp

data = []
csv_header = ["class1", "class2", "class3", "class4", "class5"]
date = []
dt = datetime.datetime(2017, 1, 10, 14, 0, 0)
end = datetime.datetime(2024, 6, 10, 13, 0, 0)
step = datetime.timedelta(days=1)

while dt < end:
    date.append(dt.strftime('%Y-%m-%d'))
    dt += step

with open(f'{FILE_PATH}.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for i, row in enumerate(rows):
    if i == 0:
      header = row        
    else:
      data.append(row)      # Sort data by ascending order of time

# calculate profit and loss
data = str2float(data)
class1 = calc_pl(data, 1)
class2 = calc_pl(data, 2)
class3 = calc_pl(data, 3)
class4 = calc_pl(data, 4)
class5 = calc_pl(data, 5)

# calculate sharpe ratio
sharpe1 = calc_sharpe(class1)
sharpe2 = calc_sharpe(class2)
sharpe3 = calc_sharpe(class3)
sharpe4 = calc_sharpe(class4)
sharpe5 = calc_sharpe(class5)

# one hot encoding
one_hot = one_hot_encoding(sharpe1, sharpe2, sharpe3, sharpe4, sharpe5)

f = pd.DataFrame(one_hot, columns=csv_header)
f.insert(0, 'time', date)
f.to_csv(f"{OUTPUT_PATH}/BTC_label.csv", index_label=False, index=False)



