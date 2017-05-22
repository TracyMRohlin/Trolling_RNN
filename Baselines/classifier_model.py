import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import confusionmatrix
from sklearn import metrics

# Read the data into a Pandas dataframe.
comments = pd.read_csv('attack_annotated_comments.tsv', sep='\t', index_col=0)
annotations = pd.read_csv('attack_annotations.tsv',  sep='\t')

# Label a comment as an attack if over half of annotators did so.
# We can tinker with this threshold later.
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# Join labels and comments
comments['attack'] = labels

tknz = TweetTokenizer()

print('tokenizing...')
# Preprocess the data -- remove newlines, tabs, quotes
# Something to consider: remove Wikipedia style markup (::'s and =='s)
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", " "))
comments['comment'] = comments['comment'].apply(lambda x: tknz.tokenize(x))

# Use a small slice for testing
# train_data = comments.iloc[0:100]
# test_data = comments.iloc[200:210]

# Grab the training data (seems to be 60%)
train_data = comments.loc[comments['split'] == 'train']
test_data = comments.loc[comments['split'] == 'test']


# bag-of-words features
def word_feats(words):
    return dict([(word, True) for word in words])

print('creating feature sets...')
# Create the training set and test set
train_set = list(zip([word_feats(x) for x in train_data['comment'].tolist()], train_data['attack'].tolist()))

gold_labels = test_data['attack'].tolist()
test_features = [word_feats(x) for x in test_data['comment'].tolist()]
test_set = list(zip(test_features, gold_labels))

print('classifying...')
# Classify and print accuracy on test set
classifier = NaiveBayesClassifier.train(train_set)
sys_label = classifier.classify_many(test_features)

sys_prob = [classifier.prob_classify(test_features[j]).prob(True) for j in range(len(test_features))]

f1 = metrics.f1_score(gold_labels, sys_label)
accuracy = classify.accuracy(classifier, test_set)
roc = metrics.roc_auc_score(gold_labels, sys_prob)

cm = confusionmatrix.ConfusionMatrix(gold_labels, sys_label)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

print('\nAccuracy: {}\nF1: {}\nROC: {}'.format(accuracy, f1, roc))
