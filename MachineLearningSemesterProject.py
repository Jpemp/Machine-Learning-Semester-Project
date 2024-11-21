#This project involves a classification of review data on a scale of 0 to 1, with 0 being negative sentiment and 1 being positive sentiment

from tkinter.messagebox import NO
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nltk.download('stopwords') #downloading stopword dictionary to prevent stopwords not found error
stop_words = set(stopwords.words('english')) #setting stopword dictionary for stopword removal later

#Loading test and train dataset into pandas dataframes
train_reviews = pd.read_csv('train.csv', delimiter=',', nrows=800000, names=['Polarity', 'Title', 'Message']) #Train data is way too big for the code to execute quickly, reducing size to 800,000
test_reviews = pd.read_csv('test.csv', delimiter=',', nrows=200000, names=['Polarity', 'Title', 'Message']) #Test data reduced to 200,000 account for train data reduction

#Display Polarity Occurrences on a Bar Graph
def polarity_bar(train_reviews, test_reviews):
    train_bar = train_reviews.Polarity.value_counts()
    print(f"Number of Polarity Occurrences: {train_bar}")
    train_bar.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Polarity Occurrences")
    plt.title("Number of times each polarity occurs in the train dataset")
    plt.show()
    plt.close()


    test_bar = test_reviews.Polarity.value_counts()
    print(f"Number of Polarity Occurrences: {test_bar}")
    test_bar.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Occurrences")
    plt.title("Number of times each polarity occurs in the test dataset")
    plt.show()
    plt.close()

#Data Preparation 

#Preprocessing data
def data_cleaning(data):
    if not isinstance(data, str):
        data = str(data) #any unforseen data that isn't a string is turned into a string data type
    data = data.lower() #turns any uppercase letters to lowercase letters
    data = data.split() #splits up data to read through by for loops
    no_stopwords = [i for i in data if i not in stop_words] #removes stopwords from data
    no_punctuation = [i for i in no_stopwords if i not in string.punctuation] #removes punctuation from data
    no_digits = " ".join([i for i in no_punctuation if not i.isdigit()]) #removes digits from data and also combines the split words back into full sentences
    return no_digits

#Term Frequency-Inverse Document Frequency
def TF_IDF(data):
    tfidf = TfidfVectorizer(lowercase=True)
    tfidf_data = tfidf.fit_transform(data)
    return tfidf_data

#Preprocessing of train and test data
#print(train_reviews)
train_reviews = train_reviews.replace(np.nan, ' ').astype(str) #gets rid of issues in reading the data
train_reviews['Polarity'] = train_reviews['Polarity'].map({'1':'0', '2':'1'}) #Changing the polarity numbers 1-2 of dataset to 0-1. Still the same polarity just represented differently
train_reviews['Processed Text'] = train_reviews['Title'] + " " + train_reviews['Message']
#print(train_reviews)
train_reviews['Processed Text'] = train_reviews['Processed Text'].apply(data_cleaning)
#print(train_reviews)

#print(test_reviews)
test_reviews = test_reviews.replace(np.nan, ' ').astype(str)  #gets rid of issues in reading the data
test_reviews['Polarity'] = test_reviews['Polarity'].map({'1':'0', '2':'1'}) #Changing the polarity numbers of dataset to 0-1. Still the same polarity just represented differently
test_reviews['Processed Text'] = test_reviews['Title'] + " " + test_reviews['Message']
#print(test_reviews)
test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(data_cleaning)
#print(test_reviews)

print(train_reviews.shape)

print(test_reviews.shape)

#Display bar graphs
polarity_bar(train_reviews, test_reviews)

#Splitting data frames into x and y values
X_train = TF_IDF(train_reviews['Processed Text'])
Y_train = train_reviews['Polarity']

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")


X_test = TF_IDF(test_reviews['Processed Text'])
Y_test = train_reviews['Polarity']

print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of Y_test: {Y_test.shape}")

def sigmoid(s):
    return 1/(1+np.exp(-s))

def sentiment_training(X_train, Y_train):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    #training_test = model.score(X_train, Y_train)  #<- to see if what i did was right
    #print("Training test: ", training_test)
    return model

def prediction(X, weight, bias):
    y_pred = sigmoid(np.dot(X, weight.T) + bias)
    return y_pred

#def model_accuracy()

def model_log_loss(y_true, y_pred): #AKA cross entropy
    loss_value = log_loss(y_true, y_pred)
    return loss_value

#def gradient_descent

#def train_model

#def test_model
