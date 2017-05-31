#! /usr/bin/env python3
# coding: utf-8

# In[1]:
import keras
import pandas as pd
import numpy as np
import urllib
import random
import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.text import Tokenizer


# In[2]:

# Download annotated comments and annotations.
# If you're Tracy, Courtney, or Amandalynne, don't run this step
# because you already have the data! If you aren't us, you will
# probably need to do this step.
# It will take a while.
ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7038044'
ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7383751'


def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)

# not needed if files already exist in directory
#download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
#download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

# Read the data into a Pandas dataframe.
comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')

# Label a comment as an attack if over half of annotators did so.
# We can tinker with this threshold later.
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# Join labels and comments
comments['attack'] = labels

# Preprocess the data -- remove newlines, tabs, quotes
# Something to consider: remove Wikipedia style markup (::'s and =='s)
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", " "))


# In[5]:

# Grab the training, dev, and test data (3:1:1 split)
train_data = comments.loc[comments['split'] == 'train']
dev_data = comments.loc[comments['split'] == 'dev']
test_data = comments.loc[comments['split'] == 'test']


# In[6]:

# The list of gold-standard labels for the data
train_labels = train_data["attack"].tolist()
dev_labels = dev_data["attack"].tolist()
test_labels = test_data["attack"].tolist()
# Put all the training data (comments) into a list
train_texts = train_data["comment"].tolist()
dev_texts = dev_data["comment"].tolist()
test_texts = test_data["comment"].tolist()


# In[7]:

def randomly_sample(comments, labels):
    # grab all the attacks to see how many we need to match
    attack_indices = [i for i in range(len(comments)) if labels[i] == True]
    new_training = [comments[i] for i in attack_indices]
    new_labels = [labels[i] for i in attack_indices]

    # grab all the ones that are not attacks, shuffle them and select the same number of non-attacks
    non_attack_indices = [i for i in range(len(comments)) if labels[i] == False]
    random.shuffle(non_attack_indices)
    for i in range(len(attack_indices)):
        new_training.append(comments[i])
        new_labels.append(labels[i])

    # shuffle the training and labeling so that they're still matched 1:1
    shuf_indices = [i for i in range(len(new_training))]
    random.shuffle(shuf_indices)
    shuffled_training = [new_training[i] for i in shuf_indices]
    shuffled_labels = [new_labels[i] for i in shuf_indices]

    return shuffled_training, shuffled_labels



# In[8]:

# randomly select non-attack data such that it creates an even split with the attack data
train_texts, train_labels = randomly_sample(train_texts, train_labels)
dev_texts, dev_labels = randomly_sample(dev_texts, dev_labels)
test_texts, test_labels = randomly_sample(test_texts, test_labels)
train_labels = np.asarray(train_labels)
dev_labels = np.asarray(dev_labels)
test_labels = np.asarray(test_labels)


# In[9]:

# The word level tokenizer using only top 5000 words
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)


# In[10]:

# Fit it to the training data.
tokenizer.fit_on_texts(train_texts)


# In[11]:

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# In[12]:

dev_tokenizer =  keras.preprocessing.text.Tokenizer(num_words=5000)
dev_tokenizer.fit_on_texts(dev_texts)

test_tokenizer =  keras.preprocessing.text.Tokenizer(num_words=5000)
test_tokenizer.fit_on_texts(test_texts)


# In[13]:

# Transform each comment in the training data to arrays of equal length
# I should really make this tokenization step into a function! To-do for AP.
train_matrix = tokenizer.texts_to_matrix(train_texts)
dev_matrix = tokenizer.texts_to_matrix(dev_texts)
test_matrix = tokenizer.texts_to_matrix(test_texts)

classifier = MultinomialNB()
classifier.fit(train_matrix, train_labels)

predictions = classifier.predict(test_matrix)

print(confusion_matrix(test_labels, predictions))
print(accuracy_score(test_labels, predictions))



