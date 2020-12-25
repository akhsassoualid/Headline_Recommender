# Libraries to import dataset, for plots 
import pandas as pd 
import numpy as np

from nltk.tokenize import word_tokenize

# libraries to build the recommender system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  NMF, LatentDirichletAllocation
from gensim.models import Word2Vec
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances



def recommender_engine(data, headline_article, n_similar_article, model):
    # Set parameters
    nc = 25
    target_ix = data[data['headline']==headline_article].index.tolist()[0]
    proc_headlines = data['headline_']
    sentence_ix = data.index.tolist() # index each headline
  
    tfidf = TfidfVectorizer()
    csr_mat = tfidf.fit_transform(proc_headlines)
    # Word embedding with Tfidf using NMF decomposition 
    if model == 'nmf':
        nmf = NMF(n_components = nc)
        features = nmf.fit_transform(csr_mat)

    # Word embedding with Tfidf using LDA decomposition
    elif model == 'lda':
        lda = LatentDirichletAllocation(n_components=nc)
        features = lda.fit_transform(csr_mat)
  
    # Word embedding using word2vec
    elif model == "word2vec":
        # We fit the model to get the weights of each word
        headline_token = [[word for word in word_tokenize(headline)] for headline in proc_headlines]
        w2v = Word2Vec(sentences=headline_token, size = nc, min_count=1, workers=3)
        vocabulary = list(w2v.wv.vocab.keys())
        
        # we compute the weights for each headline
        w2v_headline = []
        for headline in proc_headlines:
            w2v_word = np.zeros(nc, dtype='float32')
            for word in headline.split():
                if word in vocabulary:
                    w2v_word = np.add(w2v_word, w2v[word])
            w2v_word = np.divide(w2v_word, len(headline.split()))
            w2v_headline.append(w2v_word)
            features = w2v_headline

    df_features = pd.DataFrame(features, index = sentence_ix, columns = ["weights"+str(k) for k in range(nc)]) 
    target = np.array(df_features.iloc[target_ix,:]).reshape(1,-1)
    similarity = pairwise_distances(features, target, metric = "euclidean")

    df = pd.DataFrame({'category':data['category'],
                     'published':data['date'],
                     'headline': data['headline'],
                     'similarity':similarity.ravel().tolist()})
    # Show recommendations
    recom = df.nsmallest(n=n_similar_article, columns='similarity')
    print("*"*20, 'Target article', "*"*50)
    print(" "*3 + str(df['category'][target_ix])+' :', df['headline'][target_ix], " "*40)
    print("*"*20, 'Recommended articles', "*"*44)
    return recom.iloc[1:,]