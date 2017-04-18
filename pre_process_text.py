#!/usr/bin/env python3

import pandas as pd
import pickle
from collections import Counter

def main():
    # Only doing one type... remove loop entirely?
    comment_type = ["attack"]
    for c_type in comment_type:
        comments = pd.read_csv(c_type + '_annotated_comments.tsv', sep = '\t', index_col = 0)
        annotations = pd.read_csv(c_type + '_annotations.tsv',  sep = '\t')
        labels = annotations.groupby('rev_id')[c_type].mean() > 0.5
        comments[c_type] = labels
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", " "))

        comments.to_pickle(c_type + ".pkl")

if __name__ == "__main__":
    main()
