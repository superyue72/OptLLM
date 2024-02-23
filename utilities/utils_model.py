
import pandas as pd
import os
import json


def get_score_table_(df_pre_accuracy, df_llm_scores, df_llm_ability, model_list):
    score_table = []
    for i in range(len(df_pre_accuracy)):
        accuracy = []
        for model_name in model_list:
            accuracy.append(df_pre_accuracy.iloc[i][model_name] * df_llm_scores.loc[model_name]["F1-Score"] * df_llm_ability.loc[model_name]["score"])
        score_table.append(accuracy)

    score_table_df = pd.DataFrame(score_table, columns=model_list)
    return score_table_df

def get_single_model_results_(model, df_true_accuracy, df_cost):
    model_column = f'{model}'
    f_accuracy = df_true_accuracy[model_column].sum() / len(df_true_accuracy)
    f_cost = df_cost[model_column].sum()
    return [f_cost, f_accuracy]

