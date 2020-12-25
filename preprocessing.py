import pandas as pd 
import numpy as np
import json
from datetime import datetime

import string
import re
import unidecode
import nltk
from collections import Counter
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

def general_process(data, column_headline, new_column_headline):
    # Define function to process each text alone:
    def preprocess_text(text):
        stop_words = stopwords.words('english')
        lem = WordNetLemmatizer()
        # Lower Casing
        new_text = text.lower()

        # Remove stopwords and punctuations
        word_text = word_tokenize(new_text)
        punct = string.punctuation
        word_text = [word for word in word_text if not word in stop_words]
        word_text = [word for word in word_text if word.isalpha()]
        word_text = [word for word in word_text if len(word)>=2]
        word_text = [word.translate(str.maketrans('', '', punct)) for word in word_text]

        # Lemmatize the text data
        word_text = [lem.lemmatize(word) for word in word_text]
        text = ' '.join(word_text)
  
        # remove URl from text
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub(r'', text)
        return text

    # Defibe a function to remove rare words:
    def remove_rare_words(texts, n=30):
        cnt = Counter()
        for text in texts:
            for word in text.split():
                cnt[word] +=1
        rare = [w for (w, wc) in cnt.most_common()[:-n-1:-1]]
        for text in texts:
            word_text = word_tokenize(text)
            word_text = [word for word in word_text if not word in rare]
            text = ' '.join(word_text)
        print("The most rare words are : {}".format(rare))
        return texts
    data = data[data[column_headline].apply(lambda x:len(x.split())>5)]
    data[new_column_headline] = data[column_headline].apply(preprocess_text)data[new_column_headline] = remove_rare_words(data[new_column_headline], 10)
    return data