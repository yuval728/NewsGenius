import torch
import scraper
import translation, summarizer, sentiment
import joblib
import streamlit as st

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load models outside of the Streamlit functions to avoid reloading them each time the button is clicked
@st.cache_resource(show_spinner=False)
def load_models():
    language_detection_model = joblib.load('Models/language_detector.pkl')
    translation_models = translation.load_models()
    sentiment_model, vocab, tokenizer = sentiment.load_vocab_tokenizer_model(
        'Models/sentiment.pth', 'Models/vocab.pt', 100, 128, 2, 3
    )
    return language_detection_model, translation_models, sentiment_model, vocab, tokenizer

# Load models once
language_detection_model, translation_models, sentiment_model, vocab, tokenizer = load_models()

st.title("NewsGenius")
st.markdown("This app summarizes and translates news articles. It uses a combination of machine learning models to detect the language of the article, translate it to English, summarize the content, and predict the sentiment of the article.")

url = st.text_input("URL", value="", help="Enter a URL to translate and summarize")
translate_btn = st.button("Translate and Summarize")

if translate_btn:
    with st.container():
        st.title("Summary")
        
        with st.spinner("Fetching content..."):
            news_data = scraper.crawler(url)

        if news_data is None:
            st.warning("Invalid URL")
        else:
            try:
                with st.spinner("Translating and Summarizing..."):
                    language = translation.detect_language([news_data['content']], language_detection_model)
                    if language == 'English':
                        summarized_content = summarizer.generate_summary(news_data['content'], num_sentences=5)
                        translated_content = news_data['content']
                        translated_title = news_data['title']
                    elif language not in translation_models:
                        st.warning("Currently, we only support translation from Spanish, French, German, Italian, Dutch, Portuguese, Russian. Please try again with a different language.")
                        st.stop()
                    else:
                        model, tokenizer = translation_models[language]
                        translated_title = translation.translate_text(news_data['title'], model, tokenizer)
                        translated_content = translation.translate_text(news_data['content'], model, tokenizer)
                        summarized_content = summarizer.generate_summary(translated_content, num_sentences=3)
                
                output_sentiment = sentiment.sentiment_predictor(summarized_content, sentiment_model, vocab, tokenizer)
                if output_sentiment is not None:
                    c1, c2 = st.columns([1, 2], gap="small")
                    with c1:
                        st.markdown("### Sentiment: ")
                    with c2:
                        if output_sentiment == 0:
                            st.error("Negative")
                        elif output_sentiment == 1:
                            st.warning("Neutral")
                        else:
                            st.success("Positive")

                st.markdown(f"### {translated_title}")
                st.markdown(f"Original Language: {language}")
                st.markdown("---")
                st.write(summarized_content)
            except Exception as e:
                st.warning(f"Unable to translate content. Please try again. Error: {e}")
