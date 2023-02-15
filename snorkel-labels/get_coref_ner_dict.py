import pickle
from collections import defaultdict


def make_coref_ner_dict(coref_dict, ner_dict, coref_ner_dict_out_file):
    coref_ner_dict = defaultdict(dict)

    for idx, corefs in coref_dict.items():
        if idx not in ner_dict:
            print("{} not in ner dict, continueing.".format(idx))
            continue
        coref_ner_dict[idx] = []
        for cluster in corefs:
            ners = []
            for phrase in cluster:
                ner_tag = ner_dict[idx].get(phrase, None)
                ners.append(ner_tag)
            coref_ner_dict[idx].append(ners)

    # Save
    with open(coref_ner_dict_out_file, "wb") as coref_ner_dict_out_f:
        pickle.dump(coref_ner_dict, coref_ner_dict_out_f)

    return coref_ner_dict
