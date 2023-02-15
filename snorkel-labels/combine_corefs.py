import global_vars


def connect_corefs(df, coref_dict, coref_ner_dict, coref_match_id_dict):
    for doc_id, coref_clusters in coref_dict.items():

        ners = coref_ner_dict.get(doc_id, None)

        for cluster_id, coref_cluster in enumerate(coref_clusters):
            first_coref = coref_cluster[0]
            coref_matching_id_cluster = coref_match_id_dict[doc_id][cluster_id]

            main_coref, main_ids = find_main_coref(
                coref_cluster, ners, cluster_id, coref_matching_id_cluster
            )

            for coref in coref_cluster:
                if coref == first_coref or coref == main_coref:
                    continue
                elif main_coref is None or main_ids is None or coref is None:
                    continue
                else:
                    df = merge_coref_in_df(df, coref, main_coref, main_ids, doc_id)

    return df


def find_main_coref(coref_cluster, ners, cluster_id, matching_cluster):
    if ners and "PERSON" in ners[cluster_id]:
        idx = ners[cluster_id].index("PERSON")
    else:
        idx = 0

    return coref_cluster[idx], matching_cluster[idx]


def merge_coref_in_df(df, coref, main_coref, main_ids, doc_id):
    df.loc[
        (
                (df["abstract_id"] == doc_id)
                & (df["const_a"] == coref)
                & (df["labels_final"] != global_vars.WHAT)
                & (df["labels_final"] != global_vars.WHO)
        ),
        [
            "const_a",
            "sent_idx_a",
            "start_idx_a",
            "end_idx_a",
            "abstract_start_idx_a",
            "abstract_end_idx_a",
        ],
    ] = [
        main_coref,
        main_ids[0],
        main_ids[1][0],
        main_ids[1][1],
        main_ids[2][0],
        main_ids[2][1],
    ]

    df.loc[
        (
                (df["abstract_id"] == doc_id)
                & (df["const_b"] == coref)
                & (df["labels_final"] != global_vars.WHAT)
                & (df["labels_final"] != global_vars.WHO)
        ),
        [
            "const_b",
            "sent_idx_b",
            "start_idx_b",
            "end_idx_b",
            "abstract_start_idx_b",
            "abstract_end_idx_b",
        ],
    ] = [
        main_coref,
        main_ids[0],
        main_ids[1][0],
        main_ids[1][1],
        main_ids[2][0],
        main_ids[2][1],
    ]

    return df
