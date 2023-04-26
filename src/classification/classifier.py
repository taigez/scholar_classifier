import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
import trafilatura
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import numpy as np


tokenizer_e = AutoTokenizer.from_pretrained("Taige/model-edu")
model = AutoModelForSequenceClassification.from_pretrained("Taige/model-edu")
pipe_edu = pipeline(model="Taige/model-edu")
pipe_awd = pipeline(model="Taige/model-awd")
pipe_int = pipeline(model="Taige/model-int")


def get_text(url):
    print(url)
    downloaded = trafilatura.fetch_url(url)
    result = trafilatura.extract(downloaded)
    return trafilatura.utils.sanitize(result)

def multiclass(text):
    score = np.array([-1,-1,-1])
    a = pipe_awd(text)[0]
    e = pipe_edu(text)[0]
    i = pipe_int(text)[0]
    if a['label'] == 'LABEL_1':
        score[0] = a['score']
    elif e['label'] == 'LABEL_1':
        score[1] = e['score']
    elif i['label'] == 'LABEL_1':
        score[2] = i['score']
    
    if np.max(score) != -1:
        return np.argmax(score)
    else:
        return -1
    
def get_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def scrape_google(query):

    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)

    links = list(response.html.absolute_links)
    google_domains = ('https://www.google.', 
                      'https://google.', 
                      'https://webcache.googleusercontent.', 
                      'http://webcache.googleusercontent.', 
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.')

    for url in links[:]:
        # remove google domains
        if url.startswith(google_domains):
            links.remove(url)

        # remove non edu links
        elif 'linkedin' in url:
            links.remove(url)
            
    return links

def split_sen(text):
    sentences = []
    nlp = spacy.load("en_core_web_lg")
    if text != None:
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append(sent.text)
    return sentences

def scholar_search(name):
    final_dict = {}
    awd = []
    edu = []
    interest = []
    for link in scrape_google(name):
        texts = split_sen(get_text(link))
        if texts is not None:
            for t in texts:
                pred = multiclass(t)
                if pred == 0:
                    awd.append(t)
                elif pred == 1:
                    edu.append(t)
                elif pred == 2:
                    interest.append(t)
        else:
            continue
    final_dict["awd"] = awd
    final_dict["edu"] = edu
    final_dict["int"] = interest
    return final_dict

def url_search(urls):
    final_dict = {}
    awd = []
    edu = []
    interest = []
    for link in urls:
        texts = split_sen(get_text(link))
        if texts is not None:
            for t in texts:
                pred = multiclass(t)
                if pred == 0:
                    awd.append(t)
                elif pred == 1:
                    edu.append(t)
                elif pred == 2:
                    interest.append(t)
        else:
            continue
    final_dict["awd"] = awd
    final_dict["edu"] = edu
    final_dict["int"] = interest
    return final_dict