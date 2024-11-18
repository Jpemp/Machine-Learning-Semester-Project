#This project involves a classification of review data on a scale of 1 to 2, with 1 being negative and 2 being positive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading test and train dataset into pandas dataframes
train_reviews = pd.read_csv('train.csv', delimiter=',', names=["Polarity", "Title", "Message"])
test_reviews = pd.read_csv('test.csv', delimiter=',', names=["Polarity", "Title", "Message"])

#If you wish to look at the pandas dataframes for the train and test datasets, un-comment the two statements below
#print(train_reviews)
#print(test_reviews)

#Display Polarity Occurrences on Bar Graph
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
