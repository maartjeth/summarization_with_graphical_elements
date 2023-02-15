LOWERCASE = True

CLOSENESS_THRESHOLD = 2
NEIGHBOR_THRESHOLD = 1
SENT_DIS_THRESHOLD = 2

# effect-pattern-cause
# Inspired by https://www.ijcai.org/Proceedings/2020/0502.pdf
CAUSAL_PATTERNS_EPC = (
    "as a consequence of ",
    "as a result of ",
    "as long as ",
    "because ",
    "because of ",
    "caused by ",
    "due to ",
    "owing to ",
    "in response to ",
    "on account of ",
    "result from ",
    "results from ",
    "resulting from ",
    "resulted from ",
    "have resulted from ",
    "has resulted from ",
    "will result from ",
    "will have been resulted from ",
)

# cause-pattern-effect
CAUSAL_PATTERNS_CPE = (
    "accordingly ",
    "consequently ",
    "bring on ",
    "brings on ",
    "bringing on ",
    "brought on ",
    "have brought on ",
    "has brought on ",
    "will bring on ",
    "will have brought on ",
    "bring about ",
    "brings about ",
    "bringing about ",
    "brought about ",
    "have brought about ",
    "has brough about ",
    "will bring about ",
    "will have brought about ",
    "give rise to ",
    "gives rise to ",
    "giving rise to ",
    "gave rise to ",
    "have given rise to ",
    "has given rise to ",
    "will give rise to ",
    "will have given rise to ",
    "induce ",
    "induces ",
    "inducing ",
    "induced ",
    "has induced ",
    "have induced ",
    "will induce ",
    "will have induced ",
    "in order to ",
    "lead to ",
    "leads to ",
    "leading to ",
    "led to ",
    "have led to ",
    "has led to ",
    "will lead to ",
    "will have led to ",
    "result in ",
    "results in ",
    "resulting in ",
    "resulted in ",
    "will result in ",
    "will have resulted in ",
    "prevent from ",
    "prevents from ",
    "preventing from ",
    "prevented from ",
    "have prevented from ",
    "has prevented from ",
    "will prevent from ",
    "will have prevented from ",
    "stop from ",
    "stops from ",
    "stopping from ",
    "stopped from ",
    "have stopped from ",
    "has stopped from ",
    "will stop from ",
    "will have stopped from ",
    "and for this reason ",
    "cause ",
    "for the purpose of ",
    "if then ",
    "so ",
    "so that ",
    "thereby ",
    "therefore ",
    "thus ",
    "hence ",
)

LINKING_VERBS = {
    "is": True,
    "are": True,
    "am": True,
    "was": True,
    "were": True,
    "can be": True,
    "could be": True,
    "will be": True,
    "would be": True,
    "shall be": True,
    "should be": True,
    "may be": True,
    "might be": True,
    "must be": True,
    "has been": True,
    "have been": True,
    "had been": True,
}
SPECIAL_TOKENS = [
    ".",
    ",",
    "+",
    "*",
    "?",
    "^",
    "$",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "|",
    "\\",
]
TENSE_TAGS = {
    "VBD": "past",
    "VBN": "past",
    "VB": "present",
    "VBG": "present",
    "VBP": "present",
    "VBZ": "present",
    "MD": "modal",
}

ABSTAIN = 0
WHO = 1
WHAT = 2
WHAT_HAPPENED = 3
WHAT_HAPPENS = 4
WHAT_WILL_HAPPEN = 5
WHERE = 6
WHEN = 7
WHY = 8

RELATION_DICT_IDX_REL = {
    0: "ABSTAIN",
    1: "WHO",
    2: "WHAT",
    3: "WHAT_HAPPENED",
    4: "WHAT_HAPPENS",
    5: "WHAT_WILL_HAPPEN",
    6: "WHERE",
    7: "WHEN",
    8: "WHY",
}

# Heuristic
PRONOUNS = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "theirs"]


def set_paths(coref_path, coref_cluster_path, ner_path):
    global COREF_PATH
    global COREF_CLUSTER_PATH
    global NER_PATH

    COREF_PATH = coref_path
    COREF_CLUSTER_PATH = coref_cluster_path
    NER_PATH = ner_path
