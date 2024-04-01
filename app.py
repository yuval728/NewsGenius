from utils import Translate, generate_summary, Crawler
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer


german_to_english = "Helsinki-NLP/opus-mt-de-en"
german_tokenizer = MarianTokenizer.from_pretrained(german_to_english)
german_model = MarianMTModel.from_pretrained(german_to_english)

french_to_english = "Helsinki-NLP/opus-mt-fr-en"
french_tokenizer = MarianTokenizer.from_pretrained(french_to_english)
french_model = MarianMTModel.from_pretrained(french_to_english)

spanish_to_english = "Helsinki-NLP/opus-mt-es-en"
spanish_tokenizer = MarianTokenizer.from_pretrained(spanish_to_english)
spanish_model = MarianMTModel.from_pretrained(spanish_to_english)


model = None
tokenizer = None
with st.container(border=True):
    # input a link
    
    st.title("Enter a URL")
    url = st.text_input("URL", value="" , help="Enter a URL to translate and summarize")
    c1, c2, c3 = st.columns([1.5,1,2], gap="small")
    with c1:
        translate_btn = st.button("Translate and Summarize")
    with c2:
        clear_btn = st.button("Clear")
    with c3:
        model_names = ["german - english", "spanish - english" , "french - english"]
        choice = st.selectbox("Model", options=model_names, label_visibility="collapsed")
        

with st.container(border=True):
    
    # input a text
    st.title("Summary")
    if not translate_btn:
        st.warning("Please input url to translate")
    if translate_btn:
        content = Crawler(url)
        if content is None:
            st.warning("Invalid URL")
            st.stop()
        else:
            if choice == "german - english":
                model = german_model
                tokenizer = german_tokenizer
            elif choice == "spanish - english":
                model = spanish_model
                tokenizer = spanish_tokenizer
            else:    
                model = french_model
                tokenizer = french_tokenizer
            news_data = Crawler(url)
            if news_data is None:
                st.warning("Invalid URL")
                st.stop()
            # print(Translate(news_data['title'], model, tokenizer))
            translated_title = Translate(news_data['title'], model, tokenizer)
            translated_content = Translate(news_data['content'], model, tokenizer)
            summarized_content = generate_summary(translated_content)
            st.markdown(f"### {translated_title}")
            st.markdown("---")
            st.write(summarized_content)
        

            

    # # divider
    # st.markdown("---")
    # st.title("Sentiment")

    