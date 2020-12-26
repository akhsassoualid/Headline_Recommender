import numpy as np
import streamlit as st
import pandas as pd
import json
from recommender import recommender_engine
from datetime import datetime
from preprocessing import general_process


# Import the json dataset
data = pd.read_json('News_Category_Dataset_v2.json', lines=True)
data['date'] = pd.to_datetime(data.date)
data['year'] = [x.year for x in data['date']]

# Let's verify duplicates existance
duplica = data[data.duplicated('headline')==True]
data = data[data.duplicated('headline')==False]

# Subset of headlines related to the year 2018
data = data[data['year'] >= 2018]
data = data.reset_index(drop=True)
headline_data = data

#We list in the toolbox all the headlines
tup = *(headline_data['headline'][i] for i in range(len(headline_data))),

# preprocessing of headlines
headline_data = general_process(data = headline_data, column_headline="headline", new_column_headline="headline_")

# Recommending
## Title 
st.title("Articles Headline Recommnder Engine")
## Sub title
st.header('Choose an article that you heave read')
head_line = st.selectbox('Choose a headline',tup)
st.write('You selected:', head_line)

# Get the index of the chosen headline
index_head = headline_data[headline_data["headline"]==head_line].index[0]

# Toolbox to select a model
MODEL = st.selectbox(
    'Choose the model you want to use for embedding headlines',
    ('nmf', 'lda', 'word2vec'))
st.write('You selected:', MODEL)

# Toolbox to select the number of suggestion wanted
num_suggestions = st.number_input('Insert and integer number of suggestions you want',min_value=1)
st.write('The current number is ', num_suggestions)

# Display the suggestions
if st.button("Search for recommended articles headlines"):
    st.dataframe(recommender_engine(data=headline_data, index=index_head, n_similar_article=num_suggestions, model=MODEL))