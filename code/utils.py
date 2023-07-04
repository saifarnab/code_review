import pandas as pd


def calculate_confidence_score(df_without_not_enough, df_with_not_enough, add_code_df, remove_code_df, operation_df):
    combine_data = []
    for i in range(0, len(add_code_df)):
        combine_data.append([df_without_not_enough['message'].values[i], add_code_df['message'].values[i],
                             add_code_df['add_code'].values[i], add_code_df['prediction'].values[i],
                             remove_code_df['remove_code'].values[i],
                             remove_code_df['prediction'].values[i],
                             operation_df['operation'].values[i], operation_df['prediction'].values[i],
                             (add_code_df['prediction'].values[i] * remove_code_df['prediction'].values[
                                 i] * operation_df['prediction'].values[i]) ** (1 / 3)
                             ])

    for i in range(0, len(df_with_not_enough)):
        if df_with_not_enough['operation'].values[i] == 3:
            combine_data.append(
                ["", df_with_not_enough['message'].values[i], df_with_not_enough['add_code'].values[i], 0,
                 df_with_not_enough['remove_code'].values[i], 0,
                 df_with_not_enough['operation'].values[i], 0, 0
                 ])

    columns = ['original_message', 'message', 'add_code', 'add_code_prediction', 'remove_code',
               'remove_code_prediction', 'operation', 'operation_prediction', 'confidence_score']
    combine_dataframe = pd.DataFrame(combine_data, columns=columns)
    combine_dataframe = combine_dataframe.sort_values(by=['confidence_score'], ascending=False)

    combine_dataframe.to_excel('dataset_with_cs.xlsx')
