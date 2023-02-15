import pickle
from collections import defaultdict


def find_ner_ids(df, ner_dict, ner_id_dict_out_file):
    ner_idx_dict = defaultdict(dict)

    for doc_id, ners in ner_dict.items():

        ner_triple_dict = defaultdict(bool)
        ner_triple_dict_sents = defaultdict(dict)
        for _, row in df[df["abstract_id"] == doc_id].iterrows():

            if row["const_a"] in ners:
                sent_idx_a = row["sent_idx_a"]
                abstract_start_a = row["abstract_start_idx_a"]
                abstract_end_a = row["abstract_end_idx_a"]

                ner_triple = (abstract_start_a, abstract_end_a, ners[row["const_a"]])
                ner_triple_dict[ner_triple] = True
                ner_triple_dict_sents[sent_idx_a][ner_triple] = True

            if row["const_b"] in ners:
                sent_idx_b = row["sent_idx_b"]
                abstract_start_b = row["abstract_start_idx_b"]
                abstract_end_b = row["abstract_end_idx_b"]

                ner_triple = (abstract_start_b, abstract_end_b, ners[row["const_b"]])
                ner_triple_dict[ner_triple] = True
                ner_triple_dict_sents[sent_idx_b][ner_triple] = True

        ner_list = list(ner_triple_dict.keys())
        ner_list = sorted(ner_list, key=lambda x: x[0])
        ner_list = [list(ner_triple) for ner_triple in ner_list]

        ner_idx_dict[doc_id]["cross_sentence"] = ner_list
        ner_idx_dict[doc_id]["prep_within_sentence"] = ner_triple_dict_sents

    # Save
    with open(ner_id_dict_out_file, "wb") as ner_id_dict_out_f:
        pickle.dump(ner_idx_dict, ner_id_dict_out_f)

    return ner_idx_dict
