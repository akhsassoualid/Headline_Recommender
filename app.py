import numpy as np
import streamlit as st
import pandas as np
import gdown
import json
from datetime import datetime

gdown --id 1mHDG0yMs9leCjHcr6h8FX5V10EpUqBIU
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
st.dataframe(data)