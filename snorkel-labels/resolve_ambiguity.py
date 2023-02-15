import pandas as pd
import random

import global_vars


def find_relation(row):
    labels = []
    if row["labels_who"]:
        labels.append(global_vars.WHO)
    if row["labels_when"]:
        labels.append(global_vars.WHEN)
    if row["labels_what"]:
        labels.append(global_vars.WHAT)
    if row["labels_what_happens"]:
        labels.append(row["labels_what_happens"])  # as there are 3 options
    if row["labels_where"]:
        labels.append(global_vars.WHERE)
    if row["labels_why"]:
        labels.append(global_vars.WHY)

    if not labels:
        return 0  # in case of no relation
    return random.choice(labels)  # in case of multiple labels we randomly select one


def resolve_ambiguity(df, seed):
    random.seed(seed)

    final_labels = []
    for _, row in df.iterrows():
        final_labels.append(find_relation(row))

    df["labels_final"] = pd.Series(final_labels).values

    return df
