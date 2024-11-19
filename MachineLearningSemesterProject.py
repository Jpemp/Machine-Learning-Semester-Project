#This project involves a classification of review data on a scale of 1 to 2, with 1 being negative and 2 being positive

from sklearn.feature_extraction.text import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading test and train dataset into pandas dataframes
train_reviews = pd.read_csv('train.csv', delimiter=',', names=["Polarity", "Title", "Message"])
test_reviews = pd.read_csv('test.csv', names=["Polarity", "Title", "Message"])

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
#Preprocessing data
def tokenization(data):
    token = re.findall("[\w']+", data)
    print(token)
    return token

def data_cleaning(data):
    print(data)
    no_digits = "".join([i for i in data if not i.isdigit()]) #removes digits from data
    no_punctuation = "".join([i for i in no_digits if i not in string.punctuation]) #removes punctuation from data
    clean = no_punctuation.lower() #turns any uppercase letters into lowercase
    return clean

def sentiment_training(data_frame):
    print(1)

def model_accuracy():
    print(2)

def sentiment_testing():
    print(3)

#May need to tokenize before cleaning text
#print(train_reviews.head())
#train_reviews['Processed Text'] = train_reviews['Title'] + " " + train_reviews['Message']
#train_reviews['Processed Text'] = train_reviews['Processed Text'].apply(data_cleaning)
#print(train_reviews)

test_reviews['Processed Text'] = test_reviews['Title'] + " " + test_reviews['Message']
test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(tokenization)
#test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(data_cleaning)
