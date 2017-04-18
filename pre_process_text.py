#!/usr/bin/env python3

import pandas as pd
import pickle
from collections import Counter


def main():
    comment_type = ["attack"]
    comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
    annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')
    labels = annotations.groupby('rev_id')[comment_type].mean() > 0.5
    comments[comment_type] = labels
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", " "))

    comments.to_pickle("attack.pkl")

if __name__ == "__main__":
    main()
