import pandas as pd

import global_vars
import relation_extraction


def has_location_order(const, ners):
    if ners.get(const, None) == "GPE":
        return True
    return False


def starts_with_epc_order(const):
    if const.startswith(global_vars.CAUSAL_PATTERNS_EPC):
        return True

    return False


def starts_with_cpe_order(const):
    if const.startswith(global_vars.CAUSAL_PATTERNS_CPE):
        return True

    return False


def swap(row, col_dict):
    swapped_row = row

    swapped_row[col_dict["sent_idx_a"]], swapped_row[col_dict["sent_idx_b"]] = (
        row[col_dict["sent_idx_b"]],
        row[col_dict["sent_idx_a"]],
    )
    swapped_row[col_dict["const_a"]], swapped_row[col_dict["const_b"]] = (
        row[col_dict["const_b"]],
        row[col_dict["const_a"]],
    )
    swapped_row[col_dict["const_type_a"]], swapped_row[col_dict["const_type_b"]] = (
        row[col_dict["const_type_b"]],
        row[col_dict["const_type_a"]],
    )
    swapped_row[col_dict["start_idx_a"]], swapped_row[col_dict["start_idx_b"]] = (
        row[col_dict["start_idx_b"]],
        row[col_dict["start_idx_a"]],
    )
    swapped_row[col_dict["end_idx_a"]], swapped_row[col_dict["end_idx_b"]] = (
        row[col_dict["end_idx_b"]],
        row[col_dict["end_idx_a"]],
    )
    (
        swapped_row[col_dict["abstract_start_idx_a"]],
        swapped_row[col_dict["abstract_start_idx_b"]],
    ) = (row[col_dict["abstract_start_idx_b"]], row[col_dict["abstract_start_idx_a"]])
    (
        swapped_row[col_dict["abstract_end_idx_a"]],
        swapped_row[col_dict["abstract_end_idx_b"]],
    ) = (row[col_dict["abstract_end_idx_b"]], row[col_dict["abstract_end_idx_a"]])
    swapped_row[col_dict["tokens_a"]], swapped_row[col_dict["tokens_b"]] = (
        row[col_dict["tokens_b"]],
        row[col_dict["tokens_a"]],
    )

    return swapped_row


def determine_edge_direction(df, ner_dict):
    column_names = {c: i for i, c in enumerate(df.columns)}

    directed_df = []
    for idx, row in df.iterrows():
        const_a = row["const_a"]

        # When
        if row["labels_final"] == global_vars.WHEN:
            if (
                    relation_extraction.prep_month(const_a)
                    or relation_extraction.prep_year(const_a)
                    or relation_extraction.year_only(const_a)
                    or relation_extraction.month_only(const_a)
                    or relation_extraction.year(const_a)
                    or relation_extraction.month(const_a)
            ):
                directed_df.append(swap(row, column_names))
            else:
                directed_df.append(row)

        # Where
        elif row["labels_final"] == global_vars.WHERE:
            ners = ner_dict.get(row["abstract_id"], {})
            if has_location_order(const_a, ners):
                directed_df.append(swap(row, column_names))
            else:
                directed_df.append(row)

        # Why
        elif row["labels_final"] == global_vars.WHY:
            if starts_with_epc_order(const_a) or starts_with_cpe_order(const_a):
                directed_df.append(swap(row, column_names))
            else:
                directed_df.append(row)

        # All the other labels
        elif row["sent_idx_a"] < row["sent_idx_b"]:
            directed_df.append(row)
        elif row["sent_idx_b"] < row["sent_idx_a"]:
            directed_df.append(swap(row, column_names))
        elif row["start_idx_a"] > row["start_idx_b"]:
            directed_df.append(swap(row, column_names))
        else:
            directed_df.append(row)

    directed_df = pd.DataFrame(directed_df)

    return directed_df
