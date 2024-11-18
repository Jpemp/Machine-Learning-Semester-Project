#This project involves a classification of review data on a scale of 1 to 2, with 1 being negative and 2 being positive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading test and train dataset
train_reviews = pd.read_csv('train.csv', delimiter=',')
test_reviews = pd.read_csv('test.csv', delimiter=',')

#Train a model for sentiment analysis

