import numpy as np
import pandas as pd
import torch
import os
import pickle
import string
from igs.OptLLM_robust import *
from utilities.evl import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample
from sklearn.metrics import make_scorer, mean_absolute_error
from utilities.utils_model import *
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from igs.get_true_pf import *
import sys
from baselines.pymoo_lib import *
from baselines.pygmo_lib import *
from prediction.bootstrap_pred import *

if __name__ == '__main__':
    test_data_size = 0.98
    datasets = ['agnews', 'coqa', 'headlines', 'sciq', 'log_parsing']

    for dataset in datasets:
        if dataset == "log_parsing":
            model_list = ["j2_mid", "j2_ultra", "Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B",
                          "Yi_6B"]
            data_name = f"../datasets/{dataset}_word2vec.pkl"

        else:
            model_list = ['gptneox_20B', 'gptj_6B', 'fairseq_gpt_13B', 'text-davinci-002', 'text-curie-001',
                          'gpt-3.5-turbo',
                          'gpt-4', 'j1-jumbo', 'j1-grande', 'j1-large', 'xlarge', 'medium']
            data_name = f"../datasets/text_classification/{dataset}_query_word2vec.pkl"

        for itr in range(1, 11):
            save_dir = f"new_res/{dataset}/{itr}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            df_pre_accuracy, df_robust_accuracy, df_true_accuracy, df_llm_scores, df_llm_ability, df_cost, acc_table = get_predict_accuracy_(data_name, model_list, itr, alpha=1,
                                                                                                        test_size=test_data_size)
            acc_table = pd.Series(acc_table, name='acc')
            acc_table.to_csv(f"{save_dir}/acc_table.csv")

        print(f"Processing {dataset} dataset, including", len(df_pre_accuracy), "jobs")

        OptLLM_res, OptLLM_solution = OptLLM(df_robust_accuracy, df_pre_accuracy, df_true_accuracy, df_cost, model_list).run()
        OptLLM_res.to_csv(f"{save_dir}/OptLLM_res.csv")

        smsemoa_res, smsemoa_solutions = sms_emoa(df_robust_accuracy, df_true_accuracy, df_cost, model_list, termination=100).run()
        smsemoa_res.to_csv(f"{save_dir}/smsemoa_res.csv")

        nsga2_res, nsga2_solutions = nsga2(df_robust_accuracy, df_true_accuracy, df_cost, model_list, termination=100).run()
        nsga2_res.to_csv(f"{save_dir}/nsga2_res.csv")

        rnsga2_res, rnsga2_solutions = rnsga2(df_robust_accuracy, df_true_accuracy, df_cost, model_list, termination=100).run()
        rnsga2_res.to_csv(f"{save_dir}/rnsga2_res.csv")

        mopso_res, mopso_solutions = MOPSO(df_robust_accuracy, df_true_accuracy, df_cost, model_list, termination=100).run()
        mopso_res.to_csv(f"{save_dir}/mopso_res.csv")
        moead_res, moead_solutions = MOEAD(df_robust_accuracy, df_true_accuracy, df_cost, model_list,
                                           termination=100).run()
        moead_res.to_csv(f"{save_dir}/moead_res.csv")
        moeadgen_res, moeadgen_solutions = MOEADGEN(df_robust_accuracy, df_true_accuracy, df_cost, model_list, termination=100).run()
        moeadgen_res.to_csv(f"{save_dir}/moeadgen_res.csv")

        true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
        true_pt.to_csv(f"{save_dir}/true_pt.csv")
