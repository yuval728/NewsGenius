from transformers import MarianMTModel, MarianTokenizer
import joblib


helsinki_models = {
    'German': 'Helsinki-NLP/opus-mt-de-en',
    'French': 'Helsinki-NLP/opus-mt-fr-en',
    'Spanish': 'Helsinki-NLP/opus-mt-es-en',
    'Italian': 'Helsinki-NLP/opus-mt-it-en',
    'Dutch': 'Helsinki-NLP/opus-mt-nl-en',
    'Russian': 'Helsinki-NLP/opus-mt-ru-en',
    'Portugeese': 'Helsinki-NLP/opus-mt-cpp-en',
}

def load_models():
    '''
    Load the translation models
    '''
    models = {}
    for lang, model_name in helsinki_models.items():
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        models[lang] = (model, tokenizer)
    return models

def Translate(text, model, tokenizer):
    '''
    Translate the input text from source language to target language
    '''
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Perform the translation
    translation = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return ' '.join(translated_text)

def detect_language(text, model):
    '''
    Detect the language of the input text
    '''
    return model.predict(text)[0]


# def test_translation(lang_model, models):
#     import pandas as pd
#     import sacrebleu
#     from collections import defaultdict
#     blue_scores =  defaultdict(float)
#     language_scores = defaultdict(float)
#     df=pd.read_csv('test/translation_test_data.csv')
#     for i in range(len(df)):
#         text = df.loc[i, 'Source']
#         target_lang = df.loc[i, 'Reference']
#         lang = detect_language([text], lang_model)
#         language_scores[target_lang]+= (lang==df.loc[i, 'Language'])
        
#         translated_text = Translate(text, models[df.loc[i, 'Language']][0], models[df.loc[i, 'Language']][1])
#         bleu = sacrebleu.corpus_bleu(translated_text, target_lang)
#         blue_scores[target_lang] = bleu.score
        
#     print('Language detection accuracy:', sum(language_scores.values())/len(df))
#     print('Average BLEU score:', sum(blue_scores.values())/len(df))
        
#     print('All tests passed!')
    
if __name__=='__main__':
    import time
    
    lang_detection_model = joblib.load('Models/language_detector.pkl')
    
    startq = time.time()
    
    models=load_models()
    
    test_translation(lang_detection_model, models)
    # texst=['Hello, how are you?', 'Bonjour, comment ça va?', 'Hola, ¿cómo estás?', 'Hallo, wie geht es dir?', 'Ciao, come stai?', 'Hallo, hoe gaat het?', 'Привет, как дела?', 'Olá, como você está?']
    # # English, French, Spanish, German, Italian, Dutch, Russian, Portuguese
    
    # for text in texst:
    #     lang= detect_language([text], lang_detection_model)
        
    #     if lang=='English':
    #         print('The text is in English')
    #     else:
    #         print('The text is in', lang)
    #         model, tokenizer = models[lang]
    #         translated_text = Translate(text, model, tokenizer)
    #         print('Translated text:', translated_text)


    print('Time taken:', time.time()-startq)
    

     
    