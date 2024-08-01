import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
import networkx as nx


def summarizer(text, num_sentences=3):
    stopWords = set(stopwords.words("english"))              
    words = word_tokenize(text)      
    freqTable = dict()                 
    for word in words:               
        word = word.lower()                 
        if word in stopWords:                 
            continue                  
        if word in freqTable:                       
            freqTable[word] += 1            
        else:          
            freqTable[word] = 1      
            
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():                 
                if sentence in sentenceValue:                     
                    sentenceValue[sentence] += freq                
                else:                     
                    sentenceValue[sentence] = freq
                    
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
        
    average = int(sumValues / len(sentenceValue))
    
    summary = ''
    for sentence in sentences:
        if sentence in sentenceValue and sentenceValue[sentence] > (1.2 * average):
            summary += " " + sentence
            
    return summary


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

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

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
        if i <= len(ranked_sentences):
            summary.append(ranked_sentences[i][1])

    return ' '.join(summary)

if __name__=='__main__':
    
    # nltk.download('stopwords')
    # nltk.download('punkt')
    
    text = '''
    The Israeli army apparently pulled tanks from the complex of the Al-Shifa hospital in the Gaza Strip. A journalist from the AFP news agency, who stayed near the clinic, observed in the morning how tanks and vehicles left the site. The military confirmed the withdrawal. The troops had left the area after a "precise operational activity".A doctor told AFP that more than 20 corpses had been recovered. Some of them had been overrun by moving vehicles.The ministry of health in the Gaza Strip, which was controlled by the militant Islamist Hamas, spoke of 300 dead bodies.In addition, there were very large damage to the property.The AP news agency told residents that the site had been "destroyed completely". Several buildings had been burned down. Hundreds of people had returned to the vicinity of the clinic in the morning and found corpses in the building and in the immediate vicinity.A resident reported that the hospital still had patients, employees and people who had sought refuge there before the war
    '''
    
    summary = generate_summary(text)
    print(summary)
    