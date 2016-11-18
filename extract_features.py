import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
from datetime import datetime

df = pd.read_csv('test.csv')

with open('test_feature03.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    colume = [['author', 'channel', 'img_num', 'video_num', 'link_num',
               'paragraph_num', 'words_num', 'words/paragraph', 'tweet_num',
               'topic_num', 'month', 'week_day', 'till_today', 'hour', 
               'if number in title']]
    writer = csv.writer(csvfile)
    writer.writerows(colume)

# train: 27643, test: 11847
for i in range(11847):
    features = []
    soup = BeautifulSoup(df.values[i][1], "html.parser")

    # Author
    author = re.findall(r'/\S*/([^(]*)/"', str(soup.head.a))
    if author:
        author = author[0]
    if not author:
        author = soup.head.div.span.span.string[3:].lower()
        author = author.replace(' ', '-')
    features.append(author)

    # Channel
    features.append(soup.body.article['data-channel'])

    # imgs number
    img = soup.find_all('img')
    features.append(len(img))

    # videos number
    video = soup.find_all('iframe')
    features.append(len(video))

    # links number
    link = soup.find_all('a')
    features.append(len(link))

    # paragraphs number
    paragraph = len(soup.find_all('p'))
    features.append(paragraph)

    # words count
    words_count = len(soup.get_text().split())
    features.append(words_count)

    # words / paragraph
    features.append(int(words_count / paragraph))

    # tweets number
    tweet = soup.find_all('blockquote')
    features.append(len(tweet))

    # topics number
    features.append(len(soup.body.footer.find_all('a')))

    # month
    try:
        features.append(soup.time['datetime'][8:11])
    except:
        features.append('N')

    # week day
    try:
        features.append(soup.time['datetime'][0:3])
    except:
        features.append('N')

    # days till today
    try:
        date_format = "%d %b %Y"
        day = soup.time['datetime'][5:16]
        d0 = datetime.strptime(day, date_format)
        d1 = datetime.strptime('01 Jan 2015', date_format)
        delta = d1 - d0
        features.append(delta.days)
    except:
        features.append('N')

    # hour
    try:
        features.append('x' + soup.time['datetime'][17:19])
    except:
        features.append('N')

    # if number in title
    features.append(int(any(char.isdigit() for char in soup.body.h1.get_text())))

    with open('test_feature03.csv', 'a', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])
