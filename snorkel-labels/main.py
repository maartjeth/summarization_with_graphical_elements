import argparse
import benepar
import neuralcoref
import os
import pandas as pd
import spacy

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from allennlp.predictors.predictor import Predictor

import candidate_selection
import filter_doubles
import global_vars
import match_coref_ids
import get_ner_ids
import get_coref_ner_dict
import nlp_pipeline

import utils

# Take care of the Spacy models
NLP = spacy.load("en_core_web_md")
SP = spacy.load("en_core_web_sm")
if spacy.__version__.startswith("2"):
    NLP.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    NLP.add_pipe("benepar", config={"model": "benepar_en3"})

neuralcoref.add_to_pipe(NLP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str)
    parser.add_argument("--coref_out_file", type=str)
    parser.add_argument("--coref_cluster_out_file", type=str)
    parser.add_argument("--coref_match_id_out_file", type=str)
    parser.add_argument("--ner_out_file", type=str)
    parser.add_argument("--ner_id_out_file", type=str)
    parser.add_argument("--coref_ner_out_file", type=str)
    parser.add_argument("--candidate_pairs_file", type=str)
    parser.add_argument("--relations_file", type=str)
    parser.add_argument("--output_data_file_default", type=str)
    parser.add_argument("--output_data_file_dygiepp_within_sentence", type=str)
    parser.add_argument("--output_data_file_dygiepp_cross_sentence", type=str)

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--default_save", action="store_true")
    parser.add_argument("--dygiepp_save_within_sentence", action="store_true")
    parser.add_argument("--dygiepp_save_cross_sentence", action="store_true")
    parser.add_argument("--run_nlp_pipeline", action="store_true")
    parser.add_argument("--get_candidate_pairs", action="store_true")
    parser.add_argument("--make_matching_dict", action="store_true")
    parser.add_argument("--make_ner_id_dict", action="store_true")
    parser.add_argument("--make_coref_ner_dict", action="store_true")
    parser.add_argument("--get_snorkel_relations", action="store_true")
    parser.add_argument("--postprocess_snorkel_df", action="store_true")
    parser.add_argument("--save_final_df", action="store_true")
    parser.add_argument("--save_to_final_json", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Make the dirs
    if not os.path.exists(os.path.dirname(args.coref_out_file)):
        os.makedirs(os.path.dirname(args.coref_out_file))
    if not os.path.exists(os.path.dirname(args.coref_cluster_out_file)):
        os.makedirs(os.path.dirname(args.coref_cluster_out_file))
    if not os.path.exists(os.path.dirname(args.coref_match_id_out_file)):
        os.makedirs(os.path.dirname(args.coref_match_id_out_file))
    if not os.path.exists(os.path.dirname(args.ner_out_file)):
        os.makedirs(os.path.dirname(args.ner_out_file))
    if not os.path.exists(os.path.dirname(args.ner_id_out_file)):
        os.makedirs(os.path.dirname(args.ner_id_out_file))
    if not os.path.exists(os.path.dirname(args.candidate_pairs_file)):
        os.makedirs(os.path.dirname(args.candidate_pairs_file))
    if not os.path.exists(os.path.dirname(args.relations_file)):
        os.makedirs(os.path.dirname(args.relations_file))
    if not os.path.exists(os.path.dirname(args.output_data_file_default)):
        os.makedirs(os.path.dirname(args.output_data_file_default))
    if not os.path.exists(
            os.path.dirname(args.output_data_file_dygiepp_within_sentence)
    ):
        os.makedirs(os.path.dirname(args.output_data_file_dygiepp_within_sentence))
    if not os.path.exists(
            os.path.dirname(args.output_data_file_dygiepp_cross_sentence)
    ):
        os.makedirs(os.path.dirname(args.output_data_file_dygiepp_cross_sentence))

    # Read in data
    print("Reading in the data...")
    df_abstracts = utils.read_data(args.input_data_file, debug=args.debug)

    # Run NLP pipeline
    predictor_coref = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
    predictor_ner = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
    )
    if args.run_nlp_pipeline:
        print("Running NLP pipeline...")
        coref_dict, coref_cluster_dict, ner_dict = nlp_pipeline.nlp_pipeline(
            df_abstracts,
            predictor_coref,
            NLP,
            args.coref_out_file,
            args.coref_cluster_out_file,
            args.ner_out_file,
        )
    else:
        print("Loading coref dict, coref cluster dict and ner dict...")
        coref_dict = utils.load_json_dicts(args.coref_out_file)
        coref_cluster_dict = utils.load_json_dicts(args.coref_cluster_out_file)
        ner_dict = utils.load_json_dicts(args.ner_out_file)

    global_vars.set_paths(
        args.coref_out_file, args.coref_cluster_out_file, args.ner_out_file
    )

    import resolve_ambiguity
    import relation_extraction
    import combine_corefs
    import edge_direction
    import filter_overlap

    # Get / load candidate pairs
    if args.get_candidate_pairs:
        print("Getting candidate pairs...")
        df_candidate_pairs = candidate_selection.get_constituents(df_abstracts, NLP)
        utils.save_df(df_candidate_pairs, args.candidate_pairs_file)
    elif os.path.exists(args.candidate_pairs_file):
        print("Loading candidate pairs...")
        df_candidate_pairs = utils.load_df(args.candidate_pairs_file)

    # Get coref id matches from candidate df / load from existing file
    if args.make_matching_dict:
        print("Getting coref match ids...")
        coref_match_dict = match_coref_ids.find_corresponding_coref_ids(
            df_candidate_pairs,
            coref_dict,
            coref_cluster_dict,
            args.coref_match_id_out_file,
        )
    elif os.path.exists(args.coref_match_id_out_file):
        print("Loading coref match ids dict...")
        coref_match_dict = utils.load_json_dicts(args.coref_match_id_out_file)

    # Get NER tags ids from candidate df / load from existing file
    if args.make_ner_id_dict:
        print("Getting ner ids....")
        ner_id_dict = get_ner_ids.find_ner_ids(
            df_candidate_pairs, ner_dict, args.ner_id_out_file
        )
    elif os.path.exists(args.ner_id_out_file):
        print("Loading NER ID dict...")
        ner_id_dict = utils.load_pickle_dicts(args.ner_id_out_file)

    # Get Coref ner dict / load from existing file
    if args.make_coref_ner_dict:
        print("Making coref ner dict...")
        coref_ner_dict = get_coref_ner_dict.make_coref_ner_dict(
            coref_dict, ner_dict, args.coref_ner_out_file
        )
    elif os.path.exists(args.coref_ner_out_file):
        print("Loading coref ner dict...")
        coref_ner_dict = utils.load_pickle_dicts(args.coref_ner_out_file)

    # Relation extraction
    if args.get_snorkel_relations:
        # Who
        print("Relation extraction -- WHO ...")
        lfs_who = [relation_extraction.who_coref_ner]
        applier_who = PandasLFApplier(lfs=lfs_who)
        L_relations_who = applier_who.apply(df=df_candidate_pairs)
        df_candidate_pairs["labels_who"] = pd.Series(L_relations_who.squeeze()).values

        # When
        print("Relation extraction -- WHEN ...")
        lfs_when = [
            relation_extraction.when_prep_month,
            relation_extraction.when_month_only,
            relation_extraction.when_prep_year,
            relation_extraction.when_year_only,
            relation_extraction.when_ner,
        ]

        applier_when = PandasLFApplier(lfs=lfs_when)
        print("Apply labeling function...")
        L_relations_when = applier_when.apply(df=df_candidate_pairs)

        print("Make label model...")
        label_model_when = LabelModel(cardinality=8, verbose=True)
        print("Fit label model...")
        label_model_when.fit(
            L_train=L_relations_when, n_epochs=500, log_freq=100
        )

        print("Predict when...")
        pred_labels_when = label_model_when.predict(L=L_relations_when)

        print("Add labels to dataframe....")
        df_candidate_pairs["labels_when"] = pd.Series(pred_labels_when).values

        # What
        print("Relation extraction -- WHAT ...")
        lfs_what = [relation_extraction.what_np_is_np]

        applier_what = PandasLFApplier(lfs=lfs_what)
        L_relations_what = applier_what.apply(df=df_candidate_pairs)

        df_candidate_pairs["labels_what"] = pd.Series(L_relations_what.squeeze()).values

        # What happens / happened / will happen
        print("Relation extraction -- WHAT HAPPENS ...")
        lfs_what_happens = [relation_extraction.what_happens_np_vp]

        applier_what_happens = PandasLFApplier(lfs=lfs_what_happens)
        L_relations_what_happens = applier_what_happens.apply(df=df_candidate_pairs)

        df_candidate_pairs["labels_what_happens"] = pd.Series(
            L_relations_what_happens.squeeze()
        ).values

        # Where
        print("Relation extraction -- WHERE ...")
        lfs_where = [
            relation_extraction.where_ner,
            relation_extraction.where_pp_ner,
            relation_extraction.where_only_loc_ner,
        ]

        applier_where = PandasLFApplier(lfs=lfs_where)
        L_relations_where = applier_where.apply(df=df_candidate_pairs)

        label_model_where = LabelModel(cardinality=8, verbose=True)
        label_model_where.fit(
            L_train=L_relations_where, n_epochs=500, log_freq=100
        )

        pred_labels_where = label_model_where.predict(L=L_relations_where)

        df_candidate_pairs["labels_where"] = pd.Series(pred_labels_where).values

        # Why
        print("Relation extraction -- WHY ...")
        lfs_why = [relation_extraction.why_epc_cpe]

        applier_why = PandasLFApplier(lfs=lfs_why)
        L_relations_why = applier_why.apply(df=df_candidate_pairs)

        df_candidate_pairs["labels_why"] = pd.Series(L_relations_why.squeeze()).values

    if args.postprocess_snorkel_df:
        # Resolving ambiguity
        print("Resolving ambiguity...")
        df_unambiguous = resolve_ambiguity.resolve_ambiguity(
            df_candidate_pairs, args.seed
        )

        # Determine edge direction
        print("Determining edge direction...")
        df_directed = edge_direction.determine_edge_direction(df_unambiguous, ner_dict)

        # Group and filter overlapping constituents
        print("Grouping and filtering overlapping constituents...")
        df_filtered = filter_overlap.filter_pairs(df_directed)

        # Combine corefs
        print("Combining coreferences...")
        df_connected_corefs = combine_corefs.connect_corefs(
            df_filtered, coref_dict, coref_ner_dict, coref_match_dict
        )

        # Delete doubles
        print("Deleting doubles...")
        df_final = filter_doubles.delete_doubles(df_connected_corefs)

    if args.save_final_df:
        # Save df with relations
        print("Saving df...")
        utils.save_df(df_final, args.relations_file)
    elif os.path.exists(args.relations_file):
        print("Loading final df...")
        df_final = utils.load_df(args.relations_file)

    if args.save_to_final_json:

        # Save final representation
        print("Save final representation...")
        if args.default_save:
            print("Default saving...")
            utils.save_to_final(
                df_final, args.output_data_file_default
            )
        if args.dygiepp_save_within_sentence:
            print("Saving Dygiepp within sentence...")
            utils.save_to_final_dygiepp_within_sentence(
                df_final,
                ner_id_dict,
                coref_dict,
                coref_match_dict,
                args.output_data_file_dygiepp_within_sentence,
            )
        if args.dygiepp_save_cross_sentence:
            print("Saving Dygiepp cross sentence...")
            utils.save_to_final_dygiepp_cross_sentence(
                df_final,
                ner_id_dict,
                coref_dict,
                coref_match_dict,
                args.output_data_file_dygiepp_cross_sentence,
            )

    print("Done!")
