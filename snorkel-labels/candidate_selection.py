import nltk
import pandas as pd
import re

import global_vars
import utils


def parse_abstract(abstract, nlp):
    processed_abstract = []
    tok_abstract = []
    constituents = []
    cum_snt_len = []
    prev = 0

    for sent_idx, sent in enumerate(abstract):
        s = list(nlp(sent).sents)
        if s:
            s = s[0]  # this takes a little longer, but ensures that the sentences
            # are parsed correctly
        else:
            print(
                "No sentence found in parse abstract, candidate_selection.py line 25."
            )
            print(abstract)
            print(s)
            print("\n")

            constituents.append([])
            processed_abstract.append("")
            tok_abstract.append([])
            cum_snt_len.append(prev)

            continue

        parse_string = s._.parse_string
        tree = nltk.Tree.fromstring(parse_string)
        tokens = tree.leaves()
        s = " ".join(tokens)

        cum_snt_len.append(prev + len(tokens))

        constituents = traverse_tree(tree, constituents, s, sent_idx, tokens, prev)
        prev = cum_snt_len[-1]

        processed_abstract.append(s)
        tok_abstract.append(tokens)

    return constituents, processed_abstract, tok_abstract


def traverse_tree(tree, constituents, sentence, sent_idx, tokens, prev):
    tag = tree.label()
    leaves = tree.leaves()

    if tag and leaves and check_validity(tag, leaves):
        const = process_const(" ".join(leaves))

        try:
            search = re.search(const, sentence)
            start_idx = search.start()
            end_idx = search.end()

            tok_start, tok_end = find_tok_idx(leaves, tokens)
            constituents.append(
                (
                    sent_idx,
                    const,
                    tag,
                    start_idx,
                    end_idx,
                    tok_start + prev,
                    tok_end + prev,
                    leaves,
                )
            )

        except Exception as e:
            print("candidate_selection.py, line 81")
            print("Exception for sentence: {}".format(sentence))
            print("Exception for const: {}".format(const))
            print(e)

    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree, constituents, sentence, sent_idx, tokens, prev)

    return constituents


def check_validity(tag, leaves):
    if tag == "S":
        return False
    if (tag == "IN" or tag == "DT" or tag == "CC") and len(leaves) == 1:
        return False
    if leaves[0] in global_vars.SPECIAL_TOKENS:
        return False

    return True


def find_tok_idx(sublst, lst, sent_idx=None, sent_len_dict=None):
    len_sublst = len(sublst)

    for ind in (i for i, e in enumerate(lst) if e == sublst[0]):
        if lst[ind: ind + len_sublst] == sublst:
            start = ind
            end = ind + len_sublst - 1

            len_before = 0

            return start + len_before, end + len_before


def process_const(const):
    # Escape all special characters
    const = re.sub("\$", "\$", const)
    const = re.sub("\^", "\^", const)
    const = re.sub("\<", "\<", const)
    const = re.sub("\>", "\>", const)
    const = re.sub("\*", "\*", const)
    const = re.sub("\+", "\+", const)
    const = re.sub("\?", "\?", const)
    const = re.sub(r"\\", r"\\", const)
    const = re.sub("\[", "\[", const)
    const = re.sub("\.", "\.", const)
    const = re.sub("\{", "\{", const)
    const = re.sub("\(", "\(", const)
    const = re.sub("\)", "\)", const)
    const = re.sub("\|", "\|", const)
    const = re.sub("\]", "\]", const)
    const = re.sub("\}", "\}", const)

    return const


def find_all_cand_pairs(single_cands, abstract, tok_abstract, abstract_id):
    all_cands = []

    for i, cand_i in enumerate(single_cands[:-1]):
        new_cand_pairs = []
        for cand_j in single_cands[i + 1:]:
            if not utils.same_range(cand_i, cand_j) and not utils.large_distance(
                    cand_i[0], cand_j[0]
            ):
                close = utils.determine_closeness(
                    cand_i[0],
                    cand_j[0],
                    (cand_i[3], cand_i[4]),
                    (cand_j[3], cand_j[4]),
                    global_vars.CLOSENESS_THRESHOLD,
                )

                cand_a, cand_b = order_candidates(cand_i, cand_j)
                cand_tuple = (
                        (abstract_id, abstract, tok_abstract) + cand_a + cand_b + (close,)
                )
                new_cand_pairs.append(cand_tuple)

        all_cands += new_cand_pairs

    return all_cands


def order_candidates(cand_a, cand_b):
    sent_idx_a = cand_a[0]
    sent_idx_b = cand_b[0]

    # Depending on the order of the sentences
    if sent_idx_a < sent_idx_b:
        return cand_a, cand_b
    if sent_idx_b < sent_idx_a:
        return cand_b, cand_a

    start_idx_a = cand_a[3]
    start_idx_b = cand_b[3]

    end_idx_a = cand_a[4]
    end_idx_b = cand_b[4]

    if start_idx_a < start_idx_b:
        return cand_a, cand_b
    elif start_idx_b < start_idx_a:
        return cand_b, cand_a
    elif end_idx_a < end_idx_b:
        return cand_a, cand_b
    else:
        return cand_b, cand_a


def get_constituents(df, nlp):
    new_df = []

    for idx, row in df.iterrows():
        sentences = utils.process_abstract(
            row["text"],
            lowercase=global_vars.LOWERCASE,
        )
        if sentences[0] == "":
            print("candidate_selection.py, line 162, sentence 0 is empty")
            print("Sentences: {}".format(sentences))
            print("Continuing for idx: {}".format(idx))
            continue
        single_cands, processed_abstract, tok_abstract = parse_abstract(sentences, nlp)
        candidates = find_all_cand_pairs(
            single_cands, processed_abstract, tok_abstract, row["id"]
        )
        new_df += candidates

    new_df = pd.DataFrame(
        new_df,
        columns=[
            "abstract_id",
            "abstract",
            "tok_abstract",
            "sent_idx_a",
            "const_a",
            "const_type_a",
            "start_idx_a",
            "end_idx_a",
            "abstract_start_idx_a",
            "abstract_end_idx_a",
            "tokens_a",
            "sent_idx_b",
            "const_b",
            "const_type_b",
            "start_idx_b",
            "end_idx_b",
            "abstract_start_idx_b",
            "abstract_end_idx_b",
            "tokens_b",
            "close",
        ],
    )

    return new_df
