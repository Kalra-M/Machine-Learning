import re
import requests
from bs4 import BeautifulSoup

#HTML tag regex
html= r"<[^>]*>"

def clean(txt):
    '''
    Return filing after replacing spaces and new lines with ''
    '''
    txt = txt.strip()
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('&nbsp;', ' ').replace('\xa0',' ')
    return txt

def fetch_filing(url):
    '''
    For fetching filings using url
    '''
    print('Fetching ', url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    txt = str(soup)
    txt = re.sub(html,' ',txt) #remove HTML tags
    txt = clean(txt)
    return txt