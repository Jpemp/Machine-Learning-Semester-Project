#This project involves a classification of review data on a scale of 0 to 1, with 0 being negative sentiment and 1 being positive sentiment

from functools import total_ordering
from tkinter.messagebox import NO
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import model_selection
from sklearn import datasets
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('stopwords') #downloading stopword dictionary to prevent stopwords not found error
stop_words = set(stopwords.words('english')) #setting stopword dictionary for stopword removal later

#Loading test and train dataset into pandas dataframes
train_reviews = pd.read_csv('train.csv', delimiter=',', nrows=8000, names=['Polarity', 'Title', 'Message']) #Train data is way too big for the code to execute quickly, reducing size to 800,000
test_reviews = pd.read_csv('test.csv', delimiter=',', nrows=2000, names=['Polarity', 'Title', 'Message']) #Test data reduced to 200,000 account for train data reduction

#Display Polarity Occurrences on a Bar Graph
def polarity_graphs(train_reviews, test_reviews):
    train = train_reviews.value_counts()
    print(f"Number of Polarity Occurrences:\n{train}")
    train.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Polarity Occurrences")
    plt.title("Number of times each polarity occurs in the train dataset")
    plt.show()
    plt.close()

    plt.pie(train, labels=train.index, autopct='%1.1f%%')
    plt.title("Distribution of Sentiment of Train Dataset")
    plt.show()
    plt.close()

    test = test_reviews.value_counts()
    print(f"Number of Polarity Occurrences:\n{test}")
    test.plot.bar()
    plt.xlabel("Polarity")
    plt.ylabel("Number of Occurrences")
    plt.title("Number of times each polarity occurs in the test dataset")
    plt.show()
    plt.close()

    plt.pie(test, labels=test.index, autopct='%1.1f%%')
    plt.title("Distribution of Sentiment of Train Dataset")
    plt.show()
    plt.close()

#Preprocessing data
def preprocessing(data):
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
train_reviews['Processed Text'] = train_reviews['Processed Text'].apply(preprocessing)
#print(train_reviews)

#print(test_reviews)
test_reviews = test_reviews.replace(np.nan, ' ').astype(str)  #gets rid of issues in reading the data
test_reviews['Polarity'] = test_reviews['Polarity'].map({'1':'0', '2':'1'}) #Changing the polarity numbers of dataset to 0-1. Still the same polarity just represented differently
test_reviews['Processed Text'] = test_reviews['Title'] + " " + test_reviews['Message']
#print(test_reviews)
test_reviews['Processed Text'] = test_reviews['Processed Text'].apply(preprocessing)
#print(test_reviews)

def sigmoid(s):
    #formula from slides
    return 1 / (1 + np.exp(-s))

def pred(X, weight, bias):
    y_pred = sigmoid(np.dot(X.toarray(), weight.T) + bias)
    return y_pred

def model_accuracy(y_true, y_pred):
    accurate_pred = 0
    total = len(y_true)
    for i in range(len(y_true)):
        if y_true.iloc[i] == y_pred[i]:
            accurate_pred = accurate_pred + 1
    accuracy = accurate_pred/total
    return accuracy

def cross_entropy(y_true, y_pred): #AKA log loss
    loss_value = log_loss(y_true, y_pred)
    return loss_value

def test_model(x_test, weight, bias):
    y_pred = pred(x_test, weight, bias)
    prediction = np.where(y_pred > 0.5, 1, 0) #if prediction is greater than 0.5, classification is 1 (positive sentiment), else its 0 (negative sentiment)
    
    prediction = prediction.tolist()
    s_prediction = [str(x)[1:-1] for x in prediction] #turns interger array into string array

    return s_prediction

X = pd.concat([test_reviews['Processed Text'], train_reviews['Processed Text']]) #combining data for TFIDF and logistic regression function
Y = pd.concat([test_reviews['Polarity'], train_reviews['Polarity']])
X_TFIDF = TF_IDF(X)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_TFIDF, Y, test_size = 0.2, shuffle=False) #splits data. KEEP test size percentage same as train.csv and test.csv row percentage

print(f"X_train size: {X_train.shape}")
print(f"Y_train size: {Y_train.shape}")

print(f"X_test size: {X_test.shape}")
print(f"Y_test size: {Y_test.shape}")

polarity_graphs(Y_train, Y_test)

model = LogisticRegression()
model.fit(X_train, Y_train) #creates a logistic regression model based on X_train and Y_train

weight = model.coef_
print(f"Weight: {weight}")
bias = model.intercept_
print(f"Bias: {bias}")
y_pred = test_model(X_test, weight, bias)
accuracy = model_accuracy(Y_test, y_pred) #compares test polarity vs predicted polarity and tells how accurate it is
print(f"Accuracy: {accuracy}") #Accuracy
ce = cross_entropy(Y_test, y_pred) #Cost(loss)
print(f"Loss: {ce}")
precision = precision_score(Y_test, y_pred, average='weighted')
print(f"Precision: {precision}")
recall = recall_score(Y_test, y_pred, average='weighted')
print(f"Recall: {recall}")
f1 = f1_score(Y_test, y_pred, average='weighted')
print(f"F1 Score: {f1}")

#confusion matrix which shows false predictions and true predictions
cmatrix = confusion_matrix(Y_test, y_pred)
display = ConfusionMatrixDisplay(cmatrix).plot()
plt.show()
plt.close()

#ROC curve which shows how well a model operates given certain false positive rates 
RocCurveDisplay.from_estimator(model, X_test, Y_test)
plt.show()
plt.close()

#Precision-recall relationship of the model
PrecisionRecallDisplay.from_estimator(model, X_test, Y_test)
plt.show()
plt.close()
