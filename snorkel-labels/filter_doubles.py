import pandas as pd


def delete_doubles(df):
    grouped = df.groupby(["abstract_id"])
    filtered_rows = []

    filtered_df = None

    for name, group in grouped:
        # Remove duplicated in group
        filtered_group = group.loc[
            group.astype(str)
                .drop_duplicates(
                subset=[
                    "const_a",
                    "start_idx_a",
                    "end_idx_a",
                    "sent_idx_a",
                    "const_b",
                    "start_idx_b",
                    "sent_idx_b",
                    "labels_final",
                ]
            )
                .index
        ]
        filtered_rows.append(filtered_group)

    filtered_df = pd.concat(pd.DataFrame(df_group) for df_group in filtered_rows)

    return filtered_df
