from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

from bs4 import BeautifulSoup
import requests




# This part need not be used here, could directly instantiated in the ui itself
model_name = "Helsinki-NLP/opus-mt-de-en"  # English to German translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [token.lower() for token in sent1 if token.lower() not in stopwords]
    sent2 = [token.lower() for token in sent2 if token.lower() not in stopwords]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def generate_summary(text, num_sentences=3):
    '''
    Generate a summary of the input text using the TextRank algorithm
    '''
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    summary = []
    for i in range(num_sentences):
        summary.append(ranked_sentences[i][1])

    return ' '.join(summary)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def translate(text, model, tokenizer, source_language="en", target_language="de"):
    '''
    Translate the input text from source language to target language
    '''
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Translate the text
    translation = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=512, num_beams=4, early_stopping=True)

    # Decode the translated text
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return ' '.join(translated_text)

def Crawler(url='https://www.indiatoday.in/india/story/arvind-kejriwals-enforcement-directorate-custody-ends-today-to-appear-in-court-2521567-2024-04-01'):
    '''
    To be implemented
    '''
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    # main=soup.find('main')
    news_title=soup.find('h1').text
    # news_subtitle=soup.find('h2').text
    news_content_box=soup.find('article')
    
    news_content=' '.join([p.text for p in news_content_box.find_all('p')])
    
    # print(news_title)
    # print(news_subtitle)
    # print()
    print(news_content)
    return {
        'title': news_title,
        # 'subtitle': news_subtitle,
        'content': news_content
    }
    


if __name__ == "__main__":
    news_data=Crawler('https://www.lepoint.fr/societe/disparition-d-emile-apres-la-decouverte-d-ossements-quelles-sont-les-pistes-des-enqueteurs-01-04-2024-2556475_23.php')
    
    #https://www.tagesschau.de/ausland/asien/israel-gaza-al-schifa-100.htmll #*German
    #https://www.dw.com/es/ej%C3%A9rcito-israel%C3%AD-se-retira-del-hospital-al-shifa-en-gaza-tras-dos-semanas-de-asedio/a-68713144 #*Spanish
    #https://www.lepoint.fr/societe/disparition-d-emile-apres-la-decouverte-d-ossements-quelles-sont-les-pistes-des-enqueteurs-01-04-2024-2556475_23.php #*French
    
    
    print(news_data['title'])
    print(translate(news_data['title'], model, tokenizer))
    # print(translate(news_data['subtitle'], model, tokenizer))
    
    # text = '''Der schnelle Braunfuchs springt über den faulen Hund. 
    # Der Hund ist faul und der Fuchs ist schnell. Der schnelle Braunfuchs springt über den faulen Hund. 
    # Der Hund ist faul und der Fuchs ist schnell.'''
    translated_text = translate(news_data['content'], model, tokenizer)
    print(translated_text)
    summary = generate_summary(translated_text)
    print(summary)
    
