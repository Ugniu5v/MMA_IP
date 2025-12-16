import pandas as pd


DF_TO_SORT = "image_features.csv"
DF_EXAMPLE = "properties.csv"


if __name__ =="__main__":
    df_to_sort = pd.read_csv(DF_TO_SORT, header=None)
    df_example = pd.read_csv(DF_EXAMPLE, header=None)

    order = pd.Categorical(df_to_sort[0], categories=df_example[0].drop_duplicates().tolist(), ordered=True)
    df_sorted = df_to_sort.assign(_order=order).sort_values("_order").drop(columns="_order")
    df_sorted.to_csv(DF_TO_SORT, header=False, index=False)