import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
from datetime import datetime

df = pd.read_csv('train.csv')

with open('train_feature07.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    colume = [['title']]
    writer = csv.writer(csvfile)
    writer.writerows(colume)

# train: 27643, test: 11847
for i in range(27643):
    targets = ['robin williams', 'world cup', 'ebola', 'malaysia airlines', 'flappy fird', 'ice bucket challenge',  'isis', 'ferguson', 'frozen', 'ukraine',\
               'paul walker', 'boston marathon bombing', 'nelson mandela', 'cory monteith', 'iphone', 'government shutdown', 'james gandolfini', 'harlem shake', 'royal baby', 'adrian peterson']
    titles = []
    soup = BeautifulSoup(df.values[i][2], "html.parser")
    # get title
    j = 0
    title = soup.body.h1.get_text().lower()
    for target in targets:
        if target in title:
            j = 1

    with open('train_feature07.csv', 'a', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[j]])
