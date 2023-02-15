import json
import numpy as np


def find_overlap(coref_ids, df_ids):
    overlap = []
    smallest_distance_tpl = None
    smallest_distance = np.inf

    for ids in df_ids:

        abstract_ids = ids[0]
        distance = np.abs(abstract_ids[0] - coref_ids[0])

        if distance < smallest_distance:
            smallest_distance = distance
            smallest_distance_tpl = ids
        if abstract_ids[0] <= coref_ids[1] and coref_ids[0] <= abstract_ids[1]:
            overlap.append(ids)

    if len(overlap) == 1:  # simply return the overlap
        return overlap[0]
    elif len(overlap) > 1:  # return the phrase with most overlap
        range_coref_ids = list(range(coref_ids[0], coref_ids[1]))
        best_tpl = None
        best_overlap_len = -np.inf

        for tpl in overlap:
            range_overlap = list(range(tpl[0][0], tpl[0][1]))
            len_overlap = len(list(set(range_coref_ids) & set(range_overlap)))
            if len_overlap > best_overlap_len:
                best_overlap_len = len_overlap
                best_tpl = tpl
        return best_tpl
    else:
        return smallest_distance_tpl


def find_df_ids(const_type, coref, df_candidate_pairs):
    # Start and end in abstract level
    abstract_start_ids = list(
        df_candidate_pairs.loc[
            df_candidate_pairs["const_{}".format(const_type)] == coref
            ]["abstract_start_idx_{}".format(const_type)]
    )
    abstract_end_ids = list(
        df_candidate_pairs.loc[
            df_candidate_pairs["const_{}".format(const_type)] == coref
            ]["abstract_end_idx_{}".format(const_type)]
    )
    list_abstract_start_end_ids = list(zip(abstract_start_ids, abstract_end_ids))

    # Start and end in sentence level
    sent_start_ids = list(
        df_candidate_pairs.loc[
            df_candidate_pairs["const_{}".format(const_type)] == coref
            ]["start_idx_{}".format(const_type)]
    )
    sent_end_ids = list(
        df_candidate_pairs.loc[
            df_candidate_pairs["const_{}".format(const_type)] == coref
            ]["end_idx_{}".format(const_type)]
    )
    list_sent_start_end_ids = list(zip(sent_start_ids, sent_end_ids))

    # Sentence ids of const
    list_sent_ids = list(
        df_candidate_pairs.loc[
            df_candidate_pairs["const_{}".format(const_type)] == coref
            ]["sent_idx_{}".format(const_type)]
    )

    set_start_end_sent_ids = set(
        list(zip(list_abstract_start_end_ids, list_sent_start_end_ids, list_sent_ids))
    )

    return set_start_end_sent_ids


def find_corresponding_coref_ids(
        df_candidate_pairs, coref_dict, coref_cluster_dict, coref_match_id_out_file
):
    coref_match_id_dict = {}
    for doc_id, coref_clusters in coref_dict.items():

        coref_match_id_dict[doc_id] = []
        coref_matches = []

        for cluster_id, coref_cluster in enumerate(coref_clusters):
            coref_cluster_matches = []
            for coref_id, coref in enumerate(coref_cluster):

                coref_ids = coref_cluster_dict[doc_id][cluster_id][coref_id]
                set_start_end_sent_ids = find_df_ids("a", coref, df_candidate_pairs)

                if not set_start_end_sent_ids:
                    set_start_end_sent_ids = find_df_ids("b", coref, df_candidate_pairs)

                overlap_ids = find_overlap(coref_ids, set_start_end_sent_ids)

                if overlap_ids:

                    coref_cluster_matches.append(
                        (overlap_ids[2], overlap_ids[1], overlap_ids[0])
                    )
                else:
                    coref_cluster_matches.append(None)

            coref_matches.append(coref_cluster_matches)

        coref_match_id_dict[doc_id] = coref_matches

    # Save
    with open(coref_match_id_out_file, "w") as coref_match_out:
        json.dump(coref_match_id_dict, coref_match_out)

    return coref_match_id_dict
