import pandas as pd
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re

##############################

# functions to handle API calls

def get_article_list(url):
    
    raw_response = requests.get(url)
    soup = BeautifulSoup(raw_response.text,'html.parser')
    
    article_spans = pd.Series(soup.find_all('a', {'class' : 'thumb-container'})).apply(str)
    urls = article_spans.str.split('href=').str[1].str.split('>').str[0].str.replace('\"','')
    
    return urls

def get_article_text(urls):
    
    out = pd.DataFrame({'title':['']*len(urls),'text':['']*len(urls)})
    
    for i in range(len(urls)):
        
        url = urls[i]
        raw_response = requests.get(url)
        soup = BeautifulSoup(raw_response.text,'html.parser')
        
        # get title
        article_title = soup.find('h1',{'itemprop' : 'headline'}).contents[0].replace('\n','').strip()
                    
        # get body
        article_body = soup.find('div',{'itemprop':'articleBody'})
        article_text = pd.Series(article_body.find_all('p')).apply(str).str.replace('<p>','').str.replace('</p>','').str.cat(sep=' ')
        
        out.loc[i,'title'] = article_title
        out.loc[i,'text'] = article_text
        
    return out
    
##############################
        
# pull data
        
womens_url = 'https://www.bodybuilding.com/category/womens-workouts'
mens_url = 'https://www.bodybuilding.com/category/workouts'

womens_articles = get_article_text(get_article_list(womens_url))
mens_articles = get_article_text(get_article_list(mens_url))

# make sure that no articles overlap
mens_articles = mens_articles[~mens_articles.title.isin(womens_articles.title)]

# export data
womens_articles.to_csv('data/womens_articles.csv')
mens_articles.to_csv('data/mens_articles.csv')





