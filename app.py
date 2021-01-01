import numpy as np
import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from headlines_recommender.recommender import recommender_engine
from headlines_recommender.preprocessing import general_process
from PIL import Image
# -----------------------------------------------------------------------------------
# Import the processed dataset and orginal data
headline_data = pd.read_csv('./datasets/processed_data.csv')
data = pd.read_json('./datasets/News_Category_Dataset_v2.json', lines=True)

# Design of the App
## Title 
# We list in the toolbox all the headlines
tup0 = *(headline_data['headline'][i] for i in range(len(headline_data))),
tup1 = (" ",)
tup = tup0 + tup1

st.title("Articles Headline Recommnder Engine")

## Add background image
image = Image.open('./static/Headlines_image.jpg')
st.image(image, caption='https://www.wallpaperflare.com', use_column_width=True)

# -----------------------------------------------------------------------------------
# Add a selectbox to the sidebar:
activities = ["EDA Charts","Propose More Headlines"]
choice = st.sidebar.selectbox('Select an Option',activities)

if choice == "EDA Charts":
    st.subheader("EXploratory Charts")
    # We import the whole data of all years
    data = pd.read_json('./Datasets/News_Category_Dataset_v2.json', lines=True)
    data['date'] = pd.to_datetime(data.date)
    data['year'] = [x.year for x in data['date']]
    # Display charts
    yr = st.number_input('Insert a year',min_value=2016)
    n_data = data[data['year']==yr]
    if st.button("Display"):
        fig, ax = plt.subplots(figsize=(30,15))
        st.write(sns.countplot(x="category", data = n_data, order=n_data["category"].value_counts().index), ax=ax)
        st.title('Distribution of headlines categories in '+str(yr))
        ax.set_xlabel('Categories', size = 30)
        ax.set_ylabel('Counting', size = 30)
        plt.xticks(rotation=90, fontsize=25)
        plt.yticks(fontsize=25)
        st.pyplot(fig)


elif choice == "Propose More Headlines":
    ## Sub title
    st.header('Choose an article that you heave read')
    head_line = st.selectbox('Choose a headline',tup)
    #st.write('You selected:', head_line)

    ## Toolbox to select a model
    # Get the index of the chosen headline
    index_head = headline_data[headline_data["headline"]==head_line].index[0]

    MODEL = st.selectbox('Choose the model you want to use for embedding headlines',
    ('nmf', 'lda', 'word2vec'))
    #st.write('You selected:', MODEL)

    ## Toolbox to select the number of suggestion wanted
    num_suggestions = st.number_input('Insert and integer number of suggestions you want',min_value=1)
    #st.write('The current number is ', num_suggestions)
    
    # Display the suggestions
    if st.button("Search for recommendations"):
        st.dataframe(recommender_engine(data=headline_data, index=index_head, n_similar_article=num_suggestions, model=MODEL))