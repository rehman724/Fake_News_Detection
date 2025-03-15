import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    text = [i for i in text if i.isalnum()]

    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    pos_tags = pos_tag(text)
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return " ".join(text)


model = pickle.load(open('model.pkl', 'rb'))
count_vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

navi=st.sidebar.radio('Navigation',['Home','Detector'])

if navi=='Home':
    st.title("Fake News Detection")
    st.image('fake_news.jpg')

else:
    st.header('News Detector')
    input_news = st.text_area("Enter the News")

    if st.button('Predict'):
        transformed_news = transform_text(input_news)

        vector_input = count_vectorizer.transform([transformed_news])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("Fake News")
        else:
            st.success("Real News")