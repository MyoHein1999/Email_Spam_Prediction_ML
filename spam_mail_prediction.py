import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# loading the saved models

spam_mail_model = pickle.load(open('spam_mail_model.sav', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))

# page title
st.title('Spam Mail Prediction')

# code for Prediction
spam_mail = ''

input_mail = st.text_area("Enter the message")
# creating a button for Prediction
    
if st.button('Spam Mail Predict'):
    # convert text to feature vectors
    input_data_features = tfidf.transform([input_mail])

    # making prediction

    prediction = spam_mail_model.predict(input_data_features)

    if (prediction[0]==1):
        spam_mail = "This mail looks like ham!"

    else:
        spam_mail = "This mail looks like spam!"
        
st.success(spam_mail)
















