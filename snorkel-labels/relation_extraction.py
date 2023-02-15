import json
import re
import spacy
from snorkel.labeling import labeling_function

import global_vars

SP = spacy.load("en_core_web_sm")

with open(global_vars.COREF_PATH, "r") as coref_f:
    COREF_DICT = json.load(coref_f)

with open(global_vars.NER_PATH, "r") as ner_f:
    NER_DICT = json.load(ner_f)

""" WHO """


@labeling_function(resources=dict(coref_dict=COREF_DICT, ner_dict=NER_DICT))
def who_coref_ner(x, coref_dict, ner_dict):
    corefs = coref_dict.get(x.abstract_id, [])
    ners = ner_dict.get(x.abstract_id, [])
    overlap = check_overlap(x.const_a, x.const_b)

    if overlap:
        return (
            global_vars.ABSTAIN
        )  # 100% same words we do not want to add a relation in between

    if (
            check_coref(x.const_a, x.const_b, corefs)
            and check_ner(x.const_a, x.const_b, ners)
            and check_const(x.const_b)
    ):
        return global_vars.WHO
    else:
        return global_vars.ABSTAIN


def check_coref(const_a, const_b, corefs):
    for cluster in corefs:

        if const_a in cluster and const_b in cluster:
            return True

    return False


def check_ner(const_a, const_b, ners):
    # ^ = xor; shouldn't both be a person, as then you don't get
    # additional information that you want to capture
    if (const_a in ners and ners[const_a] == "PERSON") ^ (
            const_b in ners and ners[const_b] == "PERSON"
    ):
        return True

    return False


def check_const(const_b):
    if const_b not in global_vars.PRONOUNS:
        return True


def check_overlap(const_a, const_b):
    if const_a == const_b:
        return True


""" WHEN """


# phrase starting with preposition and containing a month
def prep_month(string):
    match = re.search(
        r"""(?ix)
    ^(in|at|on)\s+
    (.*)
    (January|February|March|April|May|June|July|August|September|October|November|December)
    (.*)
    (\s|[.,\/#!$%\^&\*;:{}=\-_`~()])?
    $
    """,
        string,
    )

    if match:
        return True
    return False


def prep_year(string):
    match = re.search(
        r"""(?ix)
    ^(in|at|on)\s+
    (.*)
    (1|2)(\d{3})
    (.*)
    (\s|[.,\/#!$%\^&\*;:{}=\-_`~()])?
    $""",
        string,
    )
    if match:
        return True
    return False


def year_only(string):
    match = re.search(
        r"""(?ix)
    ^(1|2)(\d{3})
    (\s|[.,\/#!$%\^&\*;:{}=\-_`~()])?
    $""",
        string,
    )
    if match:
        return True
    return False


def month_only(string):
    match = re.search(
        r"""(?ix)
    ^(January|February|March|April|May|June|July|August|September|October|November|December)
    (\s|[.,\/#!$%\^&\*;:{}=\-_`~()])?
    $
    """,
        string,
    )

    if match:
        return True
    return False


def year(string):
    match = re.search(
        r"""(?ix)
    (.*)
    (1|2)(\d{3})
    (.*)
    $""",
        string,
    )
    if match:
        return True
    return False


def month(string):
    match = re.search(
        r"""(?ix)
    (.*)
    (January|February|March|April|May|June|July|August|September|October|November|December)
    (.*)
    $
    """,
        string,
    )

    if match:
        return True
    return False


def before_after(string):
    match = re.search(
        r"""(?ix)
    ^(before|after)
    (.+)
    $
    """,
        string,
    )

    if match:
        return True
    return False


def check_date_ner(const, ners):
    if const in ners and ners[const] == "DATE":
        return True

    return False


@labeling_function()
def when_prep_month(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN
    if x.close and (prep_month(x.const_a) or prep_month(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_month(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (month(x.const_a) or month(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_month_only(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (month_only(x.const_a) or month_only(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_prep_year(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (prep_year(x.const_a) or prep_year(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_year(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (year(x.const_a) or year(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_year_only(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (year_only(x.const_a) or year_only(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function(resources=dict(ner_dict=NER_DICT))
def when_ner(x, ner_dict):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    ners = ner_dict.get(x.abstract_id, [])

    if x.close and (check_date_ner(x.const_a, ners) or check_date_ner(x.const_b, ners)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


@labeling_function()
def when_before_after(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print(
            'Type error in when prep month: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
                x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    if x.close and (before_after(x.const_a) or before_after(x.const_b)):
        return global_vars.WHEN
    return global_vars.ABSTAIN


""" What """


def subordinate_clause(string):
    match = re.search(
        r"""(?ix)
    ^(who|which|that)
    (\s)
    (.+)
    $
    """,
        string,
    )

    if match and match.group(3) in global_vars.LINKING_VERBS:
        return True
    return False


def check_ner_what(const_b, ners):
    # Const B shouldn't be a PERSON for WHAT relation
    if const_b in ners and ners[const_b] == "PERSON":
        return False
    return True


def check_const_what(const_a_type, const_b_type):
    if const_a_type == "NP" and const_b_type == "NP":
        return True


def check_overlap(const_a, const_b):
    if const_a == const_b:
        return True


@labeling_function(
    resources=dict(
        coref_dict=COREF_DICT, ner_dict=NER_DICT, pronouns=global_vars.PRONOUNS
    )
)
def what_np_is_np(x, coref_dict, ner_dict, pronouns):
    corefs = coref_dict.get(x.abstract_id, [])
    ners = ner_dict.get(x.abstract_id, [])

    same_sent = x.sent_idx_a == x.sent_idx_b
    both_np = x.const_type_a == "NP" and x.const_type_b == "NP"
    prp_np = (
            x.const_type_a == "PRP" and x.const_type_b == "NP"
    )  # E.g., "She is a teacher."
    overlap = check_overlap(x.const_a, x.const_b)

    if overlap:
        return (
            global_vars.ABSTAIN
        )  # 100% same words we do not want to add relations between

    if x.const_b in pronouns:
        return global_vars.ABSTAIN

    if same_sent and (both_np or prp_np):
        link = x.abstract[x.sent_idx_a][x.end_idx_a + 1: x.start_idx_b].strip().lower()
        if link in global_vars.LINKING_VERBS or subordinate_clause(link):
            return global_vars.WHAT

    elif (
            check_coref(x.const_a, x.const_b, corefs)
            and check_const_what(x.const_type_a, x.const_type_b)
            and check_ner_what(x.const_b, ners)
    ):
        return global_vars.WHAT

    return global_vars.ABSTAIN


""" What happens / happened / will happen """


@labeling_function(resources=dict(sp=SP))
def what_happens_np_vp(x, sp):
    same_sent = x.sent_idx_a == x.sent_idx_b
    np_vp = x.const_type_a == "NP" and x.const_type_b == "VP"
    neighbors = (x.start_idx_b - x.end_idx_a) <= global_vars.NEIGHBOR_THRESHOLD

    if same_sent and np_vp and neighbors:

        label = determine_what_hps_label(x, sp)
        if label == global_vars.WHAT_HAPPENS and x.sent_idx_a == 0:  # heuristic
            return global_vars.WHAT_HAPPENED
        else:
            return label

    return global_vars.ABSTAIN


def determine_what_hps_label(x, sp):
    tense = check_tense(x.const_b, sp)
    if tense == "past":
        return global_vars.WHAT_HAPPENED
    elif tense == "present":
        return global_vars.WHAT_HAPPENS
    elif tense == "future":
        return global_vars.WHAT_WILL_HAPPEN
    else:
        return global_vars.ABSTAIN


def check_tense(string, sp):
    tokens = sp(string)
    pos_tag = tokens[0].tag_

    if pos_tag in global_vars.TENSE_TAGS:
        tense = global_vars.TENSE_TAGS[pos_tag]
        if (
                tense == "modal" and str(tokens[0]) == "will"
        ):  # because it can also be a noun
            return "future"
        return global_vars.TENSE_TAGS[pos_tag]

    return False


""" Where """


def has_location(x, ners):
    if (x.const_a in ners and ners[x.const_a] == "GPE") or (
            x.const_a in ners and ners[x.const_a] == "GPE"
    ):
        return True

    return False


def pp_has_location(x, ners):
    if (ners.get(x.const_a, None) == "GPE" and x.const_type_a == "PP") or (
            ners.get(x.const_b, None) == "GPE" and x.const_type_b == "PP"
    ):
        return True

    return False


def only_location(x, ners):
    if len(x.tokens_a) > 1 and len(x.tokens_b) > 1:
        return False

    if ners.get(x.const_a, None) == "GPE" and len(x.tokens_a) == 1:
        return True

    if ners.get(x.const_b, None) == "GPE" and len(x.tokens_b) == 1:
        return True

    return False


@labeling_function(resources=dict(ner_dict=NER_DICT))
def where_ner(x, ner_dict):
    ners = ner_dict.get(x.abstract_id, {})
    has_loc = has_location(x, ners)

    if x.close and has_loc:
        return global_vars.WHERE
    return global_vars.ABSTAIN


@labeling_function(resources=dict(ner_dict=NER_DICT))
def where_pp_ner(x, ner_dict):
    ners = ner_dict.get(x.abstract_id, {})
    pp_has_loc = pp_has_location(x, ners)

    if x.close and pp_has_loc:
        return global_vars.WHERE
    return global_vars.ABSTAIN


@labeling_function(resources=dict(ner_dict=NER_DICT))
def where_only_loc_ner(x, ner_dict):
    ners = ner_dict.get(x.abstract_id, {})
    only_loc = only_location(x, ners)

    if x.close and only_loc:
        return global_vars.WHERE
    return global_vars.ABSTAIN


""" Why """


def starts_with_epc(x):
    causal_a = x.const_a.startswith(global_vars.CAUSAL_PATTERNS_EPC)
    causal_b = x.const_b.startswith(global_vars.CAUSAL_PATTERNS_EPC)

    if causal_a or causal_b:
        return True

    return False


def starts_with_cpe(x):
    causal_a = x.const_a.startswith(global_vars.CAUSAL_PATTERNS_CPE)
    causal_b = x.const_b.startswith(global_vars.CAUSAL_PATTERNS_CPE)

    if causal_a or causal_b:
        return True

    return False


@labeling_function()
def why_epc_cpe(x):
    if type(x.const_a) != str or type(x.const_b) != str:
        print('Type error in why_epc_cpe: const_a = {} type(x.const_a) = {}, const_b: {}, type(x.const_b) = {}'.format(
            x.const_a, type(x.const_a), x.const_b, type(x.const_b)))
        return global_vars.ABSTAIN

    epc = starts_with_epc(x)
    cpe = starts_with_cpe(x)

    if x.close and (epc or cpe):
        return global_vars.WHY
    return global_vars.ABSTAIN
