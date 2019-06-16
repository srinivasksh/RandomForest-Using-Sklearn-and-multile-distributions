import csv
import random
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

## Function to remove stop words from input list
def remove_stop_words(ipList):
	opList = []
	file_processed = []
	for ix in ipList:
		sentence = sorted([w for w in list(set(ix)) if not w in set(stoplist)])
		tmp = " ".join(str(x) for x in sentence)
		opList.append(tmp)
	return opList
	
## Create a preprocessed file with features (bag of words) and class
def create_preprocessed_file(train_matrix,class_list):
	processed_file = []
	processed_file.append(",".join(str(x) for x in vocabulary))
	ix = 0
	for train_rec in train_matrix:
		tmp_rec = np.array(train_rec).flatten()
		tmp_rec = ",".join(str(x) for x in tmp_rec)
		tmp_rec = tmp_rec + "," + class_list[ix]
		processed_file.append(tmp_rec)
		ix += 1
	return processed_file

# Load Training data set
with open("traindata.txt") as f:
    traindata = [sorted(tuple(line)) for line in csv.reader(f, delimiter=" ")]
print("Number of traindata records: %d" % len(traindata))

# Load Test data set
with open("testdata.txt") as f:
    testdata = [sorted(tuple(line)) for line in csv.reader(f, delimiter=" ")]
print("Number of testdata records: %d" % len(testdata))

# Load Training labels	
f = open("trainlabels.txt","r")
trainlabels = f.read().splitlines()

# Load stoplist
f = open("stoplist.txt","r")
stoplist = f.read().splitlines()

## Remove stop words from Training data
train_processed = remove_stop_words(traindata)
test_processed = remove_stop_words(testdata)

## Define count vectorizer and transform into matrix using bag of words
pattern = "(?u)\\b[\'\w-]+\\b"
vectorizer = CountVectorizer(token_pattern=pattern)
train_data = vectorizer.fit_transform(train_processed).todense()
test_data = vectorizer.transform(test_processed)

## Create a vocabulary of words
vocabulary = vectorizer.get_feature_names()
vocabulary.append("class")

## Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(train_data, np.matrix(trainlabels).T, test_size=0.30,random_state=47)

## Binomial Naive Bayes Classifier
bnb = BernoulliNB()
bnb = bnb.fit(X_train, np.array(y_train))

## Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb = mnb.fit(X_train, np.array(y_train))

## Score on test set using Binomial Naive Bayes
test_accuracy = bnb.score(X_test, y_test)
print("Accuracy on Test data (Binomial Naive Bayes): %f " % test_accuracy)

## Score on test set using Multinomial Naive Bayes
test_accuracy = mnb.score(X_test, y_test)
print("Accuracy on Test data (Multinomial Naive Bayes): %f " % test_accuracy)
print(" ")

## Predict the class on test data
target_class = bnb.predict(test_data)

## Proprocess training and test files
train_processed_file = create_preprocessed_file(train_data,trainlabels)

# Writing preprocessed data to a file
with open('preprocessed.txt', 'w') as f:
    for item in train_processed_file:
        f.write("%s\n" % item)
f.close()

# Writing preprocessed data to a file
with open('results.txt', 'w') as f:
    for item in target_class:
        f.write("%s\n" % item)
f.close()