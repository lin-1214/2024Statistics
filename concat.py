import pandas as pd
import numpy as np
import datetime

date = []
dt = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2024, 6, 10, 23, 59, 59)
step = datetime.timedelta(hours=1)

while dt < end:
    date.append(dt.strftime('%Y-%m-%d %H'))
    dt += step

FILE_PATH = "./rawData/BTC20170101_20240609"
OUTPUT_PATH = "./outputData/BTC20170101_20240609"

# filenames
csv_names = [f"{FILE_PATH}_9.csv", f"{FILE_PATH}_8.csv", f"{FILE_PATH}_7.csv",
               f"{FILE_PATH}_6.csv", f"{FILE_PATH}_5.csv", f"{FILE_PATH}_4.csv",
               f"{FILE_PATH}_3.csv", f"{FILE_PATH}_2.csv", f"{FILE_PATH}_1.csv", f"{FILE_PATH}_0.csv"]

csvs = [pd.read_csv(name, index_col=False) for name in csv_names]

csvs[1:] = [df[1:] for df in csvs[1:]]
length = np.sum([len(df) for df in csvs])
date = date[len(date) - length:]

combined = pd.concat(csvs)
combined.insert(0, 'time', date)
combined.to_csv(f"{OUTPUT_PATH}.csv", index_label=False, index=False)