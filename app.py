import numpy as np
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from preprocessing import general_process

option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)
