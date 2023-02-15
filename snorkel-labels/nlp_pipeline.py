import json
from collections import defaultdict

import utils


def nlp_pipeline(
        df, predictor_coref, NLP, coref_out_file, coref_cluster_out_file, ner_out_file
):
    coref_dict = defaultdict(list)
    ner_dict = defaultdict(dict)
    coref_cluster_dict = defaultdict()
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print("Working on idx: {}".format(idx))
        sent_abstract = utils.process_abstract(
            row["text"],
            lowercase=False,
        )  # no lower case as otherwise the models work less well
        abstract = " ".join(sent_abstract)

        # Coreferences
        output = predictor_coref.predict(document=abstract)

        for cluster in output["clusters"]:
            coref_dict[row["id"]].append(
                (
                    [
                        " ".join(
                            output["document"][coref_ids[0]: coref_ids[1] + 1]
                        ).lower()
                        for coref_ids in cluster
                    ]
                )
            )
        coref_cluster_dict[row["id"]] = output["clusters"]

        # NER tagging
        doc = NLP(abstract)
        for ent in doc.ents:
            ner_dict[row["id"]][ent.text.lower()] = ent.label_

    # Save
    with open(coref_out_file, "w") as coref_out:
        json.dump(coref_dict, coref_out)
    with open(coref_cluster_out_file, "w") as coref_cluster_out:
        json.dump(coref_cluster_dict, coref_cluster_out)
    with open(ner_out_file, "w") as ner_out:
        json.dump(ner_dict, ner_out)

    return coref_dict, coref_cluster_dict, ner_dict
