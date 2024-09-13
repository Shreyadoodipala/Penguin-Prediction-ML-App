import streamlit as st
import pandas as pd

st.title(':penguin: Penguin Prediction ML App')
st.info('This is a ML app.')

with st.expander('Data'):
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
