#CS4395 Project1 webcrawler
#2019/10/07 Jin Chen

import urllib
from urllib import request
from bs4 import BeautifulSoup
import codecs
import re
import string
from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import stopwords
import math
from textblob import TextBlob as tb


# scrapping from URL list
def scrapText():
    with open('urls.txt', 'r') as f:
        urls = f.read()
    urls = urls.split('\n')
    urls = [x for x in urls if x] 
# Looping through the urls
    idx = 0
    for url in urls:
        idx += 1
        html = request.urlopen(url).read().decode('utf8')
        soup = BeautifulSoup(html)
        
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out   
        # extract text
        text = soup.get_text()
        texts = text.split('\n')
        resstr = ''
        for line in texts:   # Looping the sentences for each file to a new file
            if (line != ''):
                resstr += line + '\n'
        # Save into text files
        with codecs.open('out/url-' + str(idx) + '.txt', 'w', 'utf-8') as f:
            f.write(resstr)
        print('out/url-' + str(idx) + '.txt')
    
# Clean text files
def cleanText():
    idx = 0
    files = listdir('out/')
    for i in files:
        idx += 1
        file = join('out', i)
        with codecs.open(file, 'r', 'utf-8') as tf:
            text = tf.read()
        text_chunks = [chunk for chunk in text.splitlines() if not re.match(r'^\s*$', chunk)]
        chunk_str = ''
        for i, chunk in enumerate(text_chunks):
            chunk_str += chunk + ' '
        # write into new file
        with codecs.open('cleaned_out/url-' + str(idx) + '.txt', 'w', 'utf-8') as f:
            f.write(chunk_str)
        print('cleaned_out/url-' + str(idx) + '.txt')
# using tf_idf method to extract top important terms
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def getHighFrequency():
    bloblist = []
    
    # Get the stopwords with English
    stop_words = set(stopwords.words('english'))
    idx = 0
    files = listdir('cleaned_out/')
    for i in files:
        idx += 1
        file = join('cleaned_out', i)
        with codecs.open(file, 'r', 'utf-8') as tf:
            text = tf.read()
    
        text = text.lower()
        bloblist.append(tb(text))
        # Initialize the FreqDist
        FreqDist = None
        # Remove punctuation symbols before tokenizing
        no_punc_str = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
        # Extract tokens by using NLTK tokenizer
        words = nltk.tokenize.word_tokenize(no_punc_str)
        # Remove stopwords
        no_stop_words = [w for w in words if not w in stop_words]    
        # cumulative FreqDist for all files
        if (FreqDist == None):
            FreqDist = nltk.FreqDist(no_stop_words)
        else:
            FreqDist += nltk.FreqDist(no_stop_words)
        # Sort and get the most 10 common words
        sortedFDist = sorted(FreqDist , key = FreqDist.__getitem__, reverse = True)
        common = sortedFDist[:25]
        #for j in common:
        #    print (common.index(j)+1, j)
        # print('='*50)
    
    for i, blob in enumerate(bloblist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:25]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))    
        

def createDictionary():

    with codecs.open('terms.txt', 'r', 'utf-8') as tf:
        text = tf.read()
    terms = text.split('\r\n')
    terms_arr = []
    for i in terms:
        url_num = i.split('.')[0]
        term = '.'.join(i.split('.')[1:])
        terms_arr.append([url_num, term])
        
    print(terms_arr)
    
    
    term_dict = {}
    for i in terms_arr:
        val_sentences = []
        url_num = i[0]
        term = i[1]
        filename = 'out/url-' + url_num + '.txt'
        with codecs.open(filename, 'r', 'utf-8') as tf:
            text = tf.read()
            
        lines = text.split('\n')
        for l in lines:
            temp = l.lower()
            if (temp.count(term) > 0):
                val_sentences.append(l)
        
        term_dict[term] = val_sentences
    
    for key, val in enumerate(term_dict.items()):
        print(val)
        print(val[0])
        print('-'*20)
        for j in val[1]:
            print(j)
        print('='*50)
        


scrapText()
cleanText()
getHighFrequency()
createDictionary()