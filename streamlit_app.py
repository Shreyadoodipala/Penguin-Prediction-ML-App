import streamlit as st
import pandas as pd

st.title(':penguin: Penguin Prediction ML App')
st.info('This is a ML app.')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  st.dataframe(df.head(10))

  st.write('**X**')
  X = df.drop('species', axis=1)
  st.dataframe(X.head(10))

  st.write('**Y**')
  y = df['species']
  st.dataframe(y.head(10))
