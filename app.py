import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

nltk.download('punkt')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

posMapping = {
    # "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N": 'n',
    "V": 'v',
    "J": 'a',
    "R": 'r'
}


def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):

    lower_text = text.lower()

    regex2 = r'http[s]?://\S+'
    text2 = re.sub(regex2, '', lower_text)

    text2 = re.sub(r"'s\b", '', text2)
    text2 = re.sub(r"'", '', text2)

    text2 = text2.replace("-", " ")
    text_p = re.sub(r'[!"#$%&()*+,\./:;<=>?@[\\\]^_`{|}~]', ' ', text2)
    text_p = re.sub(r'\.\.\.|…', ' … ', text_p)
    tokens = word_tokenize(text_p)

    lemmatize_words_list = []
    pos_words = pos_tag(tokens)

    for word, tag in pos_words:
        pos = posMapping.get(tag[0].upper(), 'n')  # Default POS is noun
        lemmatized_word = lemmatizer.lemmatize(word, pos)
        lemmatize_words_list.append(lemmatized_word)

    return lemmatize_words_list


def load_model_and_vectorizer():
    classifier = joblib.load("classifier.pkl")
    vectorizer = joblib.load("vectorize.pkl")
    return classifier, vectorizer


st.title("Tweet Classification App")
st.write("Predict whether a tweet leans Democratic or Republican")

user_tweet = st.text_area("Enter a tweet:", placeholder="Type your tweet here..")

if st.button("Classify"):
    classifier, vector = load_model_and_vectorizer()
    processed_tweets = " ".join(process(user_tweet))

    vectorized_tweet = vector.transform([processed_tweets])
    prediction = classifier.predict(vectorized_tweet)[0]

    if prediction == 0:
        st.success("This tweet leans **Democratic**.")
    else:
        st.success("This tweet leans **Republican**.")
else:
    st.warning("Please enter a valid tweet.")
        
