import numpy as np
import pandas as pd
from sympy import Interval, Union

import global_vars


def union(data):
    """Union of a list of intervals e.g. [(1,2),(3,4)]"""
    # Based on: https://stackoverflow.com/questions/48243507/group-rows-by-overlapping-ranges
    intervals = [Interval(begin, end) for (begin, end) in data]
    u = Union(*intervals)
    return [u] if isinstance(u, Interval) else list(u.args)


def group_overlap(df, start_col_name, end_col_name, new_column_name, group_name):
    # Create a list of intervals
    df[new_column_name] = df[[start_col_name, end_col_name]].apply(list, axis=1)
    intervals = union(df[new_column_name])

    # Add a group column
    df[group_name] = df[start_col_name].apply(
        lambda x: [g for g, l in enumerate(intervals) if l.contains(x)][0]
    )

    return df, len(intervals)


def find_best_rows(df, len_group):
    all_rows = []

    for group_idx in range(len_group):
        max_length_b = -np.inf
        max_length_a = -np.inf
        final_row = None

        for idx, row in df.loc[df["group_b"] == group_idx].iterrows():

            new_length_b = row["end_idx_b"] - row["start_idx_b"]
            if new_length_b > max_length_b:
                max_length_b = new_length_b
                final_row = row
            elif new_length_b == max_length_b:  # check row_a
                new_length_a = row["end_idx_a"] - row["start_idx_a"]
                if new_length_a > max_length_a:
                    max_length_a = new_length_a
                    final_row = row

        all_rows.append(final_row)

    return all_rows


def filter_pairs(df):
    grouped = df.groupby(["abstract_id", "sent_idx_b"])
    filtered_rows = []

    for name, group in grouped:

        grouped_final_relations = group.groupby("labels_final")

        # This is ugly, but in a loop is very slow
        if global_vars.WHO in grouped_final_relations.groups:
            group_who = grouped_final_relations.get_group(global_vars.WHO)
            group_who, len_intervals_who = group_overlap(
                group_who, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(group_who, len_intervals_who)

        elif global_vars.WHAT in grouped_final_relations.groups:
            group_what = grouped_final_relations.get_group(global_vars.WHAT)
            group_what, len_intervals_what = group_overlap(
                group_what, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(group_what, len_intervals_what)

        elif global_vars.WHAT_HAPPENS in grouped_final_relations.groups:
            group_what_happens = grouped_final_relations.get_group(
                global_vars.WHAT_HAPPENS
            )
            group_what_happens, len_intervals_what_happens = group_overlap(
                group_what_happens, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(
                group_what_happens, len_intervals_what_happens
            )

        elif global_vars.WHAT_HAPPENED in grouped_final_relations.groups:
            group_what_happened = grouped_final_relations.get_group(
                global_vars.WHAT_HAPPENED
            )
            group_what_happened, len_intervals_what_happened = group_overlap(
                group_what_happened,
                "start_idx_b",
                "end_idx_b",
                "start_end_b",
                "group_b",
            )
            filtered_rows += find_best_rows(
                group_what_happened, len_intervals_what_happened
            )

        elif global_vars.WHAT_WILL_HAPPEN in grouped_final_relations.groups:
            group_what_will_happen = grouped_final_relations.get_group(
                global_vars.WHAT_WILL_HAPPEN
            )
            group_what_will_happen, len_intervals_what_will_happen = group_overlap(
                group_what_will_happen,
                "start_idx_b",
                "end_idx_b",
                "start_end_b",
                "group_b",
            )
            filtered_rows += find_best_rows(
                group_what_will_happen, len_intervals_what_will_happen
            )

        elif global_vars.WHERE in grouped_final_relations.groups:
            group_where = grouped_final_relations.get_group(global_vars.WHERE)
            group_where, len_intervals_where = group_overlap(
                group_where, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(group_where, len_intervals_where)

        elif global_vars.WHEN in grouped_final_relations.groups:
            group_when = grouped_final_relations.get_group(global_vars.WHEN)
            group_when, len_intervals_when = group_overlap(
                group_when, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(group_when, len_intervals_when)

        elif global_vars.WHY in grouped_final_relations.groups:
            group_why = grouped_final_relations.get_group(global_vars.WHY)
            group_why, len_intervals_why = group_overlap(
                group_why, "start_idx_b", "end_idx_b", "start_end_b", "group_b"
            )
            filtered_rows += find_best_rows(group_why, len_intervals_why)

    return pd.DataFrame(filtered_rows)
