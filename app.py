from utils import *
import streamlit as st


with st.container(border=True):
    # input a link
    
    st.title("Enter a URL")
    url = st.text_input("URL")
    c1, c2 = st.columns(2)
    with c1:
        translate = st.button("Translate")
    with c2:
        model_names = {"german - english" : "Helsinki-NLP/opus-mt-de-en", "spanish-english" : "Helsinki-NLP/opus-mt-es-en", "french - english": "Helsinki-NLP/opus-mt-fr-en"} 
        choice = st.selectbox("Model", options=model_names.keys() )

with st.container(border=True):
    # input a text
    st.title("Summary")
    if not translate:
        st.warning("Please input text to translate")


    # divider
    st.markdown("---")
    st.title("Sentiment")

    