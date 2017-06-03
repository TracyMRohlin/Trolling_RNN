
# coding: utf-8

# In[ ]:

import pandas as pd
import random
from nltk.tokenize import TweetTokenizer
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import confusionmatrix
from sklearn import metrics


# In[ ]:

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


# In[ ]:

# Read the data into a Pandas dataframe.
comments = pd.read_csv('attack_annotated_comments.tsv', sep='\t', index_col=0)
annotations = pd.read_csv('attack_annotations.tsv',  sep='\t')

# Label a comment as an attack if over half of annotators did so.
# We can tinker with this threshold later.
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# Join labels and comments
comments['attack'] = labels


# In[ ]:

print('tokenizing...')

tknz = TweetTokenizer()
# Preprocess the data -- remove newlines, tabs, quotes
# Something to consider: remove Wikipedia style markup (::'s and =='s)
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", " "))
comments['comment'] = comments['comment'].apply(lambda x: tknz.tokenize(x))


# In[ ]:

# Very small data set for debugging
# train_data = comments.iloc[0:100]
# test_data = comments.iloc[200:210]

# Grab the training data (seems to be 60%)
train_data = comments.loc[comments['split'] == 'train']
dev_data = comments.loc[comments['split'] == 'dev']
test_data = comments.loc[comments['split'] == 'test']


# In[ ]:

# The list of gold-standard labels for the data
train_labels = train_data["attack"].tolist()
dev_labels = dev_data["attack"].tolist()
test_labels = test_data["attack"].tolist()
# Put all the training data (comments) into a list
train_texts = train_data["comment"].tolist()
dev_texts = dev_data["comment"].tolist()
test_texts = test_data["comment"].tolist()


# In[ ]:

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


# In[ ]:

train_texts, train_labels = randomly_sample(train_texts, train_labels)
dev_texts, dev_labels = randomly_sample(dev_texts, dev_labels)
test_texts, test_labels = randomly_sample(test_texts, test_labels)


# In[ ]:

print('creating feature sets...')

# bag-of-words features
def word_feats(words):
    return dict([(word, True) for word in words])

# Create the training set and test set
train_features = [word_feats(x) for x in train_texts]
train_set = list(zip(train_features, train_labels))
test_features = [word_feats(x) for x in test_texts]
test_set = list(zip(test_features, test_labels))


# In[ ]:

print('classifying...')
# Classify and  retreive system labels
classifier = NaiveBayesClassifier.train(train_set)
sys_label = classifier.classify_many(test_features)


# In[ ]:

# Extract Attack probabilities for AUROC measure
sys_prob = [classifier.prob_classify(test_features[j]).prob(True) for j in range(len(test_features))]
roc = metrics.roc_auc_score(test_labels, sys_prob)

# Evaluation metrics
f1 = metrics.f1_score(test_labels, sys_label)
accuracy = classify.accuracy(classifier, test_set)
print('\nAccuracy: {}\nF1: {}\nROC: {}'.format(accuracy, f1, roc))

# Print confusion matrix
cm = confusionmatrix.ConfusionMatrix(test_labels, sys_label)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

