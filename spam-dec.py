from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import streamlit as st
import pandas as pd 
import pickle 

data = pd.read_csv('data/spam.csv', encoding = 'latin-1')

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

vectorize_text = vectorizer.fit_transform(data.v2)
classifier.fit(vectorize_text, data.v1)

st.write("""
	# Spam Message Prediction App

	This app predicts if a message is spam!

	Dataset obtained from https://www.kaggle.com/uciml/sms-spam-collection-dataset
	***
	""")

user_input = st.text_area("Enter first message here:", "Fine if thatåÕs the way u feel. ThatåÕs the way its gota b", height = 50)
user_input2 = st.text_area("Enter second message here:", "I HAVE A DATE ON SUNDAY WITH WILL!!", height = 50)
user_input3 = st.text_area("Enter third message here:", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", height = 50)
#transformed_input = vectorizer.fit_transform(user_input)
baseData = [user_input, user_input2, user_input3]
columnName = ["v2"]
inputdf = pd.DataFrame(data=baseData, columns = columnName)
inputdf.to_csv('out.csv')

inputdata = pd.read_csv('out.csv', encoding = 'latin-1')
vectorize_input = vectorizer.transform(inputdata.v2)

st.write("***")

for i in range(3):
	prediction = classifier.predict(vectorize_input)[i]
	if(prediction == "spam"):
		st.write("Message ",i+1," is:  **SPAM**!")
	else:
		st.write("Message ",i+1," is:  **NOT** spam!")

