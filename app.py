import numpy as np
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from preprocessing import general_process

# Import the json dataset
data = pd.read_json('News_Category_Dataset_v2.json', lines=True)
data['date'] = pd.to_datetime(data.date)
data['year'] = [x.year for x in data['date']]

# Let's verify duplicates existance
duplica = data[data.duplicated('headline')==True]
data = data[data.duplicated('headline')==False]

# Subset of data for the years of 2018
data = data[data['year'] >= 2018]
data = data.reset_index(drop=True)
headline_data = data

# preprocessing
headline_data = general_process(data = headline_data, column_headline="headline", new_column_headline="headline_")

# Recommending
## Title 
st.title("Articles Headline Recommnder Engine")
## Image
#img =Image.open('Headlines_image.jpg')
#st.image(img, caption='www.depositphotos.com')
## Headline askd from a reader
st.header('Choose an article that you heave read')
st.subheader("You can choose from the headlines of the following link \n https://drive.google.com/file/d/1mHDG0yMs9leCjHcr6h8FX5V10EpUqBIU/view?usp=sharing")
head_line = st.text_input('copy the headline you have just read to suggest you the similar ones')
