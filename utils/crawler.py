import requests
import json
import numpy as np
import random
import os
import csv
from tqdm import tqdm
from time import sleep
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

QUOTE_LEN = 24
DAY_NUM = 2718      # 2017-01-01 ~ 2024-06-10
FILE_PATH = "../rawData/BTC20170101_20240609.csv"
url = "https://min-api.cryptocompare.com/data/v2/histohour"

# if os.path.isfile(FILE_PATH):
#   os.remove(FILE_PATH)

f_config = open("./config.json")
f_headers = open("./headers.json")
config = json.load(f_config)
header_list = json.load(f_headers)

# set retry
retry_times = 20
retry_backoff_factor = 2
session = requests.Session()
retry = Retry(total=retry_times, backoff_factor = retry_backoff_factor, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries = retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

# parse input data
aggregate = config["aggregate"]
e = config["e"]
extraParams = config["extraParams"]
fsym = config["fsym"]
limit = config["limit"]
toTs = config["toTs"]
tsym = config["tsym"]
tryConversion = config["tryConversion"]

output = []
csv_header = ["high", "low", "open", "volumefrom", "volumeto", "close"]

i = 1
while i <= DAY_NUM:
    if i > 90: 
        print(toTs)
        break
    print(i)
    headers = {'user-agent': header_list[str(random.randint(0, len(header_list)-1))]}
    res = session.get(url, params={
        "fsym": fsym,
        "tsym": tsym,
        "limit": limit,
        "toTs": toTs,
        "aggregate": aggregate,
        "e": e,
        "tryConversion": tryConversion,
        "extraParams": extraParams
    }, headers=headers, timeout=10)

    if res.status_code != 200:
        print(f"[-]Error response: {res.status_code}")
        exit(1)
    rawData = res.json().get("Data", [])
    try:
        Data = rawData["Data"]
        # print(Data)
        tmp = []
        for j in range(1, 25):
            row = []
            row.append(Data[j]["high"])
            row.append(Data[j]["low"])
            row.append(Data[j]["open"])
            row.append(Data[j]["volumefrom"])
            row.append(Data[j]["volumeto"])
            row.append(Data[j]["close"])
            tmp.append(row)
        
        output.insert(-i, tmp)
        toTs = Data[0]["time"]
        i = i + 1
    except:
        print(f"[-]Error")
        print(f"[-]Dead time: {toTs}")
        break

    # sleep(random.uniform(0, 2))

with open('../rawData/BTC20170101_20240609_9.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    for rows in output:
        for row in rows:
            writer.writerow(row)




