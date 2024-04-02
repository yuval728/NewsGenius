from utils import Translate, generate_summary, Crawler, PredictSentiment
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import tensorflow as tf


german_to_english = "Helsinki-NLP/opus-mt-de-en"
german_tokenizer = MarianTokenizer.from_pretrained(german_to_english)
german_model = MarianMTModel.from_pretrained(german_to_english)

french_to_english = "Helsinki-NLP/opus-mt-fr-en"
french_tokenizer = MarianTokenizer.from_pretrained(french_to_english)
french_model = MarianMTModel.from_pretrained(french_to_english)

spanish_to_english = "Helsinki-NLP/opus-mt-es-en"
spanish_tokenizer = MarianTokenizer.from_pretrained(spanish_to_english)
spanish_model = MarianMTModel.from_pretrained(spanish_to_english)

sentiment_model = tf.keras.models.load_model('Models/sentiment_analysis.keras')

sentiments = ["Negative", "Positive","Neutral"]
sentimenent = None
model = None
tokenizer = None

st.title("News Summarizer and Translator")
with st.container(border=True):
    # input a link
    
    st.title("Enter a URL")
    url = st.text_input("URL", value="" , help="Enter a URL to translate and summarize")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        translate_btn = st.button("Translate and Summarize")
    with c2:
        model_names = ["german - english", "spanish - english" , "french - english"]
        choice = st.selectbox("Model", options=model_names, label_visibility="collapsed")
    

with st.container(border=True):
    
    # input a text
    # with c1:
    st.title("Summary")
    
    if not translate_btn:
        st.warning("Please input url to translate")
    if translate_btn:
        with st.spinner("Fetching content..."):
            news_data = Crawler(url)

        if news_data is None:
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
            if news_data is None:
                st.warning("Invalid URL")
                st.stop()
            # print(Translate(news_data['title'], model, tokenizer))
            try:
                with st.spinner("Translating and Summarizing..."):
                
                    translated_title = Translate(news_data['title'], model, tokenizer)
                    translated_content = Translate(news_data['content'], model, tokenizer)
                    summarized_content = generate_summary(translated_content)

            except:
                st.warning("Unable to translate content. Please chose the correct language to translate to.")
                st.stop()
            sentimenent = PredictSentiment(translated_title)[0]
            if sentimenent is not None:
                c1, c2 = st.columns([1,2], gap="small")
                with c1:
                    st.markdown("### Sentiment: ")
                with c2:
                    
                        if sentimenent == 0:
                            st.error("Negative")
                        elif sentimenent == 1:
                            st.success("Positive")
                        else:
                            st.warning("Neutral")

            st.markdown(f"### {translated_title}")
            st.markdown("---")
            st.write(summarized_content)

        
  
                

    # # divider
    # st.markdown("---")
    # st.title("Sentiment")

    