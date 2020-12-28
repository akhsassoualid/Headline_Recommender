import numpy as np
import streamlit as st
import pandas as pd
import json
from recommender import recommender_engine
from datetime import datetime
from preprocessing import general_process
from PIL import Image

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.beta_columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

# Import the processed dataset
headline_data = pd.read_csv('processed_data.csv')

# We list in the toolbox all the headlines
tup0 = *(headline_data['headline'][i] for i in range(len(headline_data))),
tup1 = (" ",)
tup = tup0 + tup1

# Design of the App
## Title 
st.title("Articles Headline Recommnder Engine")

## Add background image
from PIL import Image
image = Image.open('Headlines_image.jpg')
st.image(image, caption='https://www.wallpaperflare.com', use_column_width=True)

## Sub title
st.header('Choose an article that you heave read')
head_line = st.selectbox('Choose a headline',tup)
#st.write('You selected:', head_line)

# Get the index of the chosen headline
index_head = headline_data[headline_data["headline"]==head_line].index[0]

## Toolbox to select a model
MODEL = st.selectbox(
    'Choose the model you want to use for embedding headlines',
    ('nmf', 'lda', 'word2vec'))
#st.write('You selected:', MODEL)

## Toolbox to select the number of suggestion wanted
num_suggestions = st.number_input('Insert and integer number of suggestions you want',min_value=1)
#st.write('The current number is ', num_suggestions)

# preprocessing of headlines
#headline_data = general_process(data = headline_data, column_headline="headline", new_column_headline="headline_")

# Display the suggestions
if st.button("Search for recommendions"):
    st.dataframe(recommender_engine(data=headline_data, index=index_head, n_similar_article=num_suggestions, model=MODEL))