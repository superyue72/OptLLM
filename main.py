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
from baselines.random_allocation import *
from baselines.pymoo_lib import *




if __name__ == '__main__':
    itr = sys.argv[1]
    dataset = "text_classification" #, "log_parsing, text_classification"]
    # datasets = ['overruling', 'agnews', 'coqa', 'headlines', 'sciq']

    test_data_size = 0.98
    data_name = f"datasets/{dataset}_word2vec.pkl"
    # data_name = f"datasets/{dataset}/overruling_query_word2vec.pkl"

    save_dir = f"output/{dataset}_{test_data_size}_{itr}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if dataset == "text_classification":
        model_list = ['gptneox_20B', 'gptj_6B', 'fairseq_gpt_13B', 'text-davinci-002', 'text-curie-001', 'gpt-3.5-turbo',
                      'gpt-4', 'j1-jumbo', 'j1-grande', 'j1-large', 'xlarge', 'medium']
    elif dataset == "log_parsing":
        model_list = ["j2_mid", "j2_ultra", "Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B", "Yi_6B"]


    df_pre_accuracy, df_true_accuracy, df_cost, df_llm_scores, df_llm_ability = data_preprocess(data_name, model_list, itr, test_size=test_data_size)
    # df_score_table = get_score_table_(df_pre_accuracy, df_llm_scores, df_llm_ability, model_list)
    # df_pre_accuracy, df_true_accuracy, df_cost = log_preprocess(data_dir, model_list, test_size=test_data_size)

    print(f"Processing {dataset} dataset, including", len(df_pre_accuracy), "jobs")

    smsemoa_res, smsemoa_solutions = sms_emoa(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination="100").run()
    smsemoa_res.to_csv(f"{save_dir}/{dataset}_smsemoa_res.csv")
    nsga2_res, nsga2_solutions = nsga2(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination="100").run()
    nsga2_res.to_csv(f"{save_dir}/{dataset}_nsga2_res.csv")
    rnsga2_res, rnsga2_solutions = rnsga2(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination="100").run()
    rnsga2_res.to_csv(f"{save_dir}/{dataset}_rnsga2_res.csv")

    # single_model_res = {}
    # for i in range(len(model_list)):
    #     single_model_res[model_list[i]] = get_single_model_results_(model_list[i], df_true_accuracy, df_cost)

    # if os.path.exists(f"{save_dir}/{dataset}_igs_res.csv"):
    #     igs_res = pd.read_csv(f"{save_dir}/{dataset}_igs_res.csv")
    # else:
    #     igs_res, igs_solution = iterative_greedy_search_(df_pre_accuracy, df_true_accuracy, df_cost, model_list).run()
    #     igs_res.to_csv(f"{save_dir}/{dataset}_igs_res.csv")
    #
    # if os.path.exists(f"{save_dir}/{dataset}_igs2_res.csv"):
    #     igs_res2 = pd.read_csv(f"{save_dir}/{dataset}_igs2_res.csv")
    # else:
    #     igs_res2, igs_solution2 = iterative_greedy_search_(df_score_table, df_true_accuracy, df_cost, model_list).run()
    #     igs_res2.to_csv(f"{save_dir}/{dataset}_igs2_res.csv")


    if os.path.exists(f"{save_dir}/{dataset}_true_pt.csv"):
        true_pt = pd.read_csv(f"{save_dir}/{dataset}_true_pt.csv")
    else:
        true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
        true_pt.to_csv(f"{save_dir}/{dataset}_true_pt.csv")
        true_pt_solution = pd.DataFrame(true_pt_solution)
        true_pt_solution.to_csv(f"{save_dir}/{dataset}_true_pt_solution.csv")
    #
    # # if os.path.exists(f"{save_dir}/{dataset}_expected_pt.csv"):
    # #     expected_pt = pd.read_csv(f"{save_dir}/{dataset}_expected_pt.csv")
    # # else:
    # #     expected_pt, expected_pt_solution = get_expected_pareto_front(df_pre_accuracy, df_true_accuracy, df_cost, model_list).run()
    # #     expected_pt.to_csv(f"{save_dir}/{dataset}_expected_pt.csv")
    # #     expected_pt_solution = pd.DataFrame(expected_pt_solution)
    # #     expected_pt_solution.to_csv(f"{save_dir}/{dataset}_expected_pt_solution.csv")


