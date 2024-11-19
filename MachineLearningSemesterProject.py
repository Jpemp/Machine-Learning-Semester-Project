#This project involves a classification of review data on a scale of 1 to 2, with 1 being negative and 2 being positive

from tkinter.messagebox import NO
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading test and train dataset into pandas dataframes
train_reviews = pd.read_csv('train.csv', delimiter=',', names=['Polarity', 'Title', 'Message'])
test_reviews = pd.read_csv('test.csv', delimiter=',', names=['Polarity', 'Title', 'Message'])

#If you wish to look at the pandas dataframes for the train and test datasets, un-comment the two statements below
#print(train_reviews)
#print(test_reviews)

#Display Polarity Occurrences on Bar Graph
def polarity_bar(train_reviews, test_reviews):
    train_bar = train_reviews.Polarity.value_counts()
    train_bar.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Occurrences")
    plt.title("Number of times each polarity occurs in the train dataset")
    plt.show()
    plt.close()


    test_bar = test_reviews.Polarity.value_counts()
    test_bar.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Occurrences")
    plt.title("Number of times each polarity occurs in the test dataset")
    plt.show()
    plt.close()

#Train a model for sentiment analysis

#Tokenizing data
def tokenization(data):
    token = re.findall("[\w']+", data)
    return token

#Preprocessing data
def data_cleaning(data):
    no_punctuation = "".join([i for i in data if i not in string.punctuation]) #removes punctuation from data
    no_digits = "".join([i for i in no_punctuation if not i.isdigit()]) #removes digits from data
    stop_words = stopwords.words('english')
    no_stopwords = "".join([i for i in no_digits if i not in stop_words]) #removes stopwords which can get in the way of the TF-IDF since they don't provide much sentiment
    clean = no_stopwords.lower() #turns any uppercase letters into lowercase
    return clean

#Term Frequency-Inverse Document Frequency
def TF_IDF(data):
    tfidf = TfidfVectorizer()
    tfidf_data = tfidf.fit_transform(data)
    return tfidf_data

def sentiment_training(data_frame):
    print(2)

def model_accuracy():
    print(2)

def sentiment_testing():
    print(3)

#Preprocessing and tokenization of train and test data
print(train_reviews)
train_reviews = train_reviews.replace(np.nan, '').astype(str) #gets rid of issues in reading the data
train_reviews['Processed Text'] = train_reviews['Title'] + " " + train_reviews['Message']
#print(train_reviews)
train_reviews['Processed Text'] = train_reviews['Processed Text'].apply(data_cleaning)
#print(train_reviews)
train_reviews['Processed Text'] = train_reviews['Processed Text'].apply(tokenization)
print(train_reviews)

#print(test_reviews)
test_reviews = test_reviews.replace(np.nan, '').astype(str)  #gets rid of issues in reading the data
test_reviews['Processed Text'] = test_reviews['Title'] + " " + test_reviews['Message']
#print(test_reviews)
test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(data_cleaning)
#print(test_reviews)
test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(tokenization)
#print(test_reviews)

#Splitting data frames into x,y values for LS graphs
X_train = TF_IDF(train_reviews['Processed Text'])
X_test = TF_IDF(test_reviews['Processed Text'])

Y_train = train_reviews['Polarity']
Y_test = train_reviews['Polarity']

