import itertools

import ast
import json
import newlinejson
import numpy as np
import os
import pandas as pd
import pickle
from collections import defaultdict

import global_vars


def determine_closeness(sent_idx_a, sent_idx_b, ids_a, ids_b, threshold):
    if sent_idx_a != sent_idx_b:
        return False

    second_start = max(ids_a[0], ids_b[0])
    first_end = min(ids_a[1], ids_b[1])

    if second_start - first_end <= threshold:
        return True

    return False


def read_data(data_file, debug):
    with open(data_file, "rb") as of:
        d = pickle.load(of)

    df_abstracts = pd.DataFrame.from_dict(d, orient="index").reset_index()
    df_abstracts = pd.DataFrame(df_abstracts)
    df_abstracts.rename(columns={"index": "id", "sent_abstract": "text"}, inplace=True)

    if debug:
        return df_abstracts[["id", "text"]].head()
    return df_abstracts[["id", "text"]]


def process_abstract(abstract, lowercase):
    if lowercase:
        return [s.lower() for s in abstract]
    return abstract


def save_df(df, path):
    df.to_csv(path_or_buf=path, sep=";")


def load_df(path):
    df = pd.read_csv(path, sep=";", keep_default_na=False)  # to avoid NaN errors
    return df


def load_json_dicts(path):
    if os.path.exists(path):
        with open(path, "r") as out_f:
            dict = json.load(out_f)
        return dict
    else:
        raise ValueError(
            "Trying to load json dict, but {} does not exist.".format(path)
        )


def load_pickle_dicts(path):
    if os.path.exists(path):
        with open(path, "rb") as in_f:
            dict = pickle.load(in_f)
        return dict
    else:
        raise ValueError(
            "Trying to load pickle dict, but {} does not exist.".format(path)
        )


def same_range(cand_a, cand_b):
    if cand_a[0] == cand_b[0] and in_range(cand_a[3], cand_a[4], cand_b[3], cand_a[4]):
        return True

    return False


def in_range(start_a, end_a, start_b, end_b):
    return start_a < end_b and start_b < end_a


def large_distance(sent_idx_a, sent_idx_b):
    if np.abs(sent_idx_b - sent_idx_a) > global_vars.SENT_DIS_THRESHOLD:
        return True
    return False


def save_to_final(df, outf):
    output_dict = defaultdict(dict)
    grouped = df.groupby("abstract_id")

    for abs_idx, group in grouped:

        doc_dict = {}
        doc_dict["dataset"] = "cnndm"
        doc_dict["tok_abstract"] = group.iloc[0]["tok_abstract"]
        doc_dict["sent_abstract"] = group.iloc[0]["abstract"]
        doc_dict["annotations"] = {}

        # Only one here, but to match it with the label format
        ann_dict = {}
        ann_dict["triples_verbatim"] = []
        ann_dict["triples_ids_char"] = []
        ann_dict["triples_ids_tok"] = []
        ann_dict["sent_ids"] = []

        for _, row in group.iterrows():
            relation_verbatim = global_vars.RELATION_DICT_IDX_REL[
                row["labels_final"]
            ].lower()
            triple_id_char = (
                row["start_idx_a"],
                row["end_idx_a"],
                row["labels_final"],
                row["start_idx_b"],
                row["end_idx_a"],
            )
            triple_id_tok = (
                row["abstract_start_idx_a"],
                row["abstract_end_idx_a"],
                row["labels_final"],
                row["abstract_start_idx_b"],
                row["abstract_end_idx_a"],
            )
            triple_verbatim = (row["const_a"], relation_verbatim, row["const_b"])
            sent_ids = (row["sent_idx_a"], row["sent_idx_b"])

            ann_dict["triples_verbatim"].append(triple_verbatim)
            ann_dict["triples_ids_char"].append(triple_id_char)
            ann_dict["triples_ids_tok"].append(triple_id_tok)
            ann_dict["sent_ids"].append(sent_ids)

        doc_dict["annotations"][0] = ann_dict
        output_dict[abs_idx] = doc_dict

    with open(outf, "w") as out:
        json.dump(output_dict, out)


def check_validity_abs_dict(abs_dict):
    len_sents = len(
        abs_dict["sentences"][0]
    )

    if abs_dict["clusters"]:
        for cluster in abs_dict["clusters"]:
            for coref_ids in cluster:
                if max(coref_ids) >= len_sents:
                    return False

    return True


def find_tok_idx(sublst, lst):
    len_sublst = len(sublst)

    for ind in (i for i, e in enumerate(lst) if e == sublst[0]):

        if lst[ind: ind + len_sublst] == sublst:
            start = ind
            end = ind + len_sublst - 1

            return start, end

    return None, None


def save_to_final_dygiepp_within_sentence(
        df, ner_id_dict, coref_dict, coref_match_id_dict, outf
):
    grouped = df.groupby("abstract_id")

    with newlinejson.open(outf, "w") as of:

        for abs_idx, group in grouped:

            tok_abstract = ast.literal_eval(
                group["tok_abstract"].values[0]
            )

            abs_dict = {}
            abs_dict["doc_key"] = group["abstract_id"].values[0]
            abs_dict["dataset"] = "cnndm"
            abs_dict["sentences"] = tok_abstract
            abs_dict["relations"] = [[] for _ in range(len(tok_abstract))]
            abs_dict["clusters"] = []
            abs_dict["ner"] = [[] for _ in range(len(tok_abstract))]

            coref_clusters = coref_dict.get(abs_idx, [])
            coref_match_ids = coref_match_id_dict.get(abs_idx, [])
            for i, coref_cluster in enumerate(coref_clusters):
                cluster_list = []
                for j, coref in enumerate(coref_cluster):
                    match_ids = coref_match_ids[i][j]
                    if match_ids:
                        match_ids = [match_ids[2][0], match_ids[2][1]]
                        if match_ids not in cluster_list:
                            cluster_list.append(match_ids)
                abs_dict["clusters"].append(cluster_list)

            sent_len_dict = {}
            for i, s in enumerate(tok_abstract):
                sent_len_dict[i] = len(s)

            sent_accum_len_dict = {}
            current_len = 0
            for i, s in enumerate(tok_abstract):
                sent_accum_len_dict[i] = current_len
                current_len += len(s)

            # NERS
            if abs_idx in ner_id_dict:
                ner_dict = ner_id_dict[abs_idx]["prep_within_sentence"]
            else:
                ner_dict = {}
            for sent_idx, ners in ner_dict.items():
                len_so_far = sent_accum_len_dict[sent_idx]
                ners = ners.keys()
                ners_sent_idx = []
                for ner_triple in ners:
                    ner_triple_list = [
                        ner_triple[0] - len_so_far,
                        ner_triple[1] - len_so_far,
                        ner_triple[2],
                    ]
                    ners_sent_idx.append(ner_triple_list)

                ners_sent_idx = sorted(ners_sent_idx, key=lambda x: x[0])
                abs_dict["ner"][sent_idx] += ners_sent_idx

            for row_idx, row in group.iterrows():

                if row["sent_idx_a"] != row["sent_idx_b"]:
                    continue

                sent_idx = row["sent_idx_a"]  # within sentence, hence equal

                start_a, end_a = find_tok_idx(
                    row["tokens_a"],
                    row["tok_abstract"][row["sent_idx_a"]],
                )
                start_b, end_b = find_tok_idx(
                    row["tokens_b"],
                    row["tok_abstract"][row["sent_idx_b"]],
                )

                if start_a is None or start_b is None or end_a is None or end_b is None:
                    print(
                        "Continueing in Saving Dygiepp within sentence because of None encounter. Doc ID = {}, start_a = {}, end_a = {}, start_b = {}, end_b = {}. Const_a = {}, Const_b = {} ".format(
                            abs_dict["doc_key"],
                            start_a,
                            end_a,
                            start_b,
                            end_b,
                            row["tokens_a"],
                            row["tokens_b"],
                        )
                    )
                    continue

                relation_id = row["labels_final"]
                relation = global_vars.RELATION_DICT_IDX_REL[relation_id]

                abs_dict["relations"][row["sent_idx_a"]].append(
                    [start_a, end_a, start_b, end_b, relation]
                )

            of.write(abs_dict)


def save_to_final_dygiepp_cross_sentence(
        df, ner_id_dict, coref_dict, coref_match_id_dict, outf
):
    grouped = df.groupby("abstract_id")

    with newlinejson.open(outf, "w") as of:

        for abs_idx, group in grouped:

            tok_abstract = ast.literal_eval(
                group["tok_abstract"].values[0]
            )

            abs_dict = {}
            abs_dict["doc_key"] = abs_idx
            abs_dict["dataset"] = "cnndm"
            abs_dict["sentences"] = [list(itertools.chain.from_iterable(tok_abstract))]
            abs_dict["relations"] = [[]]
            abs_dict["clusters"] = []
            abs_dict["ner"] = []

            # NERS
            if abs_idx in ner_id_dict:
                abs_dict["ner"].append(ner_id_dict[abs_idx]["cross_sentence"])
            else:
                abs_dict["ner"].append([])

            # Corefs
            coref_clusters = coref_dict.get(abs_idx, [])
            coref_match_ids = coref_match_id_dict.get(abs_idx, [])
            for i, coref_cluster in enumerate(coref_clusters):
                cluster_list = []
                for j, coref in enumerate(coref_cluster):

                    match_ids = coref_match_ids[i][j]
                    if match_ids:
                        match_ids = [match_ids[2][0], match_ids[2][1]]
                        if match_ids not in cluster_list:
                            cluster_list.append(match_ids)
                abs_dict["clusters"].append(cluster_list)

            sent_len_dict = {}
            for i, s in enumerate(tok_abstract):
                sent_len_dict[i] = len(s)

            sent_accum_len_dict = {}
            current_len = 0
            for i, s in enumerate(tok_abstract):
                sent_accum_len_dict[i] = current_len
                current_len += len(
                    s
                )  # added after, so that we can use it below immediately

            # Relations
            for row_idx, row in group.iterrows():
                start_a = row["abstract_start_idx_a"]
                end_a = row["abstract_end_idx_a"]
                relation_id = row["labels_final"]
                start_b = row["abstract_start_idx_b"]
                end_b = row["abstract_end_idx_b"]

                relation = global_vars.RELATION_DICT_IDX_REL[relation_id]

                abs_dict["relations"][0].append(
                    [start_a, end_a, start_b, end_b, relation]
                )

            if check_validity_abs_dict(
                    abs_dict
            ):  # To correct for incorrectly parsed sentences
                of.write(abs_dict)
