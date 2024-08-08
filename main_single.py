#iterative greedy search for LLMs jobs allocation problem
import ast
import sys
import pandas as pd
from tqdm import tqdm
from igs.igs import iterative_greedy_search_
from prediction.prediction_model import *
from igs.get_true_pf import *
from igs.get_expected_pf import *
from utilities.utils_model import *
from baselines.single_obj_optimization import *


# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.problems import get_problem
# from pymoo.optimize import minimize
#
# problem = get_problem("g2")
#
# algorithm = GA(
#     pop_size=100,
#     eliminate_duplicates=True)
#
# res = minimize(problem,
#                algorithm,
#                seed=1,
#                verbose=False)
#
# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
#



if __name__ == '__main__':
    # itr = 2
    test_data_size = 0.98
    datasets = ['log_parsing'] #, 'sciq', 'log_parsing'] #['agnews', 'coqa', 'headlines', 'sciq', 'log_parsing'] # ["log_parsing", text_classification"]

    for dataset in datasets:
        if dataset == "log_parsing":
            model_list = ["j2_mid", "j2_ultra", "Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B",
                          "Yi_6B"]
            data_name = f"datasets/{dataset}_word2vec.pkl"

        else:
            model_list = ['gptneox_20B', 'gptj_6B', 'fairseq_gpt_13B', 'text-davinci-002', 'text-curie-001',
                          'gpt-3.5-turbo',
                          'gpt-4', 'j1-jumbo', 'j1-grande', 'j1-large', 'xlarge', 'medium']
            data_name = f"datasets/text_classification/{dataset}_query_word2vec.pkl"
        # for itr in range(1,11):
            # save_dir = f"output2/{dataset}/{itr}"
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)

        ga_pre_acc = []
        ga_true_acc = []
        ga_time = []
        de_pre_acc = []
        de_true_acc = []
        de_time = []
        pso_pre_acc = []
        pso_true_acc = []
        pso_time = []
        for itr in range(1, 10):
            # const_value = 40
            const_value = 0.45
            termination = 100
            df_pre_accuracy, df_true_accuracy, df_cost, df_llm_scores, df_llm_ability = data_preprocess(data_name, model_list, itr, test_size=test_data_size)
            df_score_table = get_score_table_(df_pre_accuracy, df_llm_scores, df_llm_ability, model_list)
            # de_res, de_accuracy, de_solution, de_elapsed_time = de_for_acc(df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination).run()
            # ga_res, ga_accuracy, ga_solution, ga_elapsed_time = ga_for_acc(df_pre_accuracy, df_true_accuracy, df_cost,
            #                                                            const_value, model_list, termination).run()
            # pso_res, pso_accuracy, pso_solution, pso_elapsed_time = pso_for_acc(df_pre_accuracy, df_true_accuracy, df_cost,
            #                                                            const_value, model_list, termination).run()
            # ga_pre_acc.append(ga_res)
            # ga_true_acc.append(ga_accuracy)
            # ga_time.append(ga_elapsed_time)
            # de_pre_acc.append(de_res)
            # de_true_acc.append(de_accuracy)
            # de_time.append(de_elapsed_time)
            # pso_pre_acc.append(pso_res)
            # pso_true_acc.append(pso_accuracy)
            # pso_time.append(pso_elapsed_time)
            pso_res, pso_accuracy, pso_solution, pso_elapsed_time = pso_for_cost(df_pre_accuracy, df_true_accuracy,
                                                                                df_cost, const_value, model_list, termination).run()

            de_res, de_accuracy, de_solution, de_elapsed_time = de_for_cost(df_pre_accuracy, df_true_accuracy, df_cost,
                                                                           const_value, model_list, termination).run()
            ga_res, ga_accuracy, ga_solution, ga_elapsed_time = ga_for_cost(df_pre_accuracy, df_true_accuracy, df_cost,
                                                                           const_value, model_list, termination).run()

            ga_pre_acc.append(ga_res)
            ga_true_acc.append(ga_accuracy)
            ga_time.append(ga_elapsed_time)
            de_pre_acc.append(de_res)
            de_true_acc.append(de_accuracy)
            de_time.append(de_elapsed_time)
            pso_pre_acc.append(pso_res)
            pso_true_acc.append(pso_accuracy)
            pso_time.append(pso_elapsed_time)
        # save the results
        df_ga_res = pd.DataFrame({"pre_acc": ga_pre_acc, "true_acc": ga_true_acc, "time": ga_time})
        df_ga_res.to_csv(f"{dataset}_ga_res.csv")
        df_de_res = pd.DataFrame({"pre_acc": de_pre_acc, "true_acc": de_true_acc, "time": de_time})
        df_de_res.to_csv(f"{dataset}_de_res.csv")
        df_pso_res = pd.DataFrame({"pre_acc": pso_pre_acc, "true_acc": pso_true_acc, "time": pso_time})
        df_pso_res.to_csv(f"{dataset}_pso_res.csv")


