#iterative greedy search for LLMs jobs allocation problem
import pickle
from tqdm import tqdm
from igs.igs import iterative_greedy_search_
from prediction.prediction_model import *
from igs.get_true_pf import *
from igs.get_expected_pf import *
from baselines.random_allocation import *


def get_single_model_results_(model, df_true_accuracy, df_cost):
    model_column = f'{model}'
    f_accuracy = df_true_accuracy[model_column].sum() / len(df_true_accuracy)
    f_cost = df_cost[model_column].sum()
    return [f_cost, f_accuracy]


if __name__ == '__main__':
    tasks = "text_classification" #, "log_parsing, text_classification"]
    datasets = ['overruling', 'agnews', 'coqa', 'headlines', 'sciq'] #'overruling', 'agnews', 'coqa', 'headlines', 'sciq']
    for dataset in datasets:
        test_data_size = 0.99
        print(f"Processing {dataset} dataset")
        data_dir = f"datasets/{tasks}"
        save_dir = f"output/text_classification/query_{test_data_size}/{dataset}_{test_data_size}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if tasks == "text_classification":
            model_list = ['gptneox_20B', 'gptj_6B', 'fairseq_gpt_13B', 'text-davinci-002', 'text-curie-001', 'gpt-3.5-turbo',
                          'gpt-4', 'j1-jumbo', 'j1-grande', 'j1-large', 'xlarge', 'medium']
        elif tasks == "log_parsing":
            model_list = ["j2_mid", "j2_ultra", "Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B", "Yi_6B"]


        df_pre_accuracy, df_true_accuracy, df_cost = data_preprocess(data_dir, dataset, model_list, test_size=test_data_size)

        # single_model_res = {}
        # for i in range(len(model_list)):
        #     single_model_res[model_list[i]] = get_single_model_results_(model_list[i], df_true_accuracy, df_cost)
        #
        # igs_res, nondominated_solutions = iterative_greedy_search_(df_pre_accuracy, df_true_accuracy, df_cost, model_list).run()
        # # if os.path.exists(f"{save_dir}/{dataset}_true_pt.csv"):
        # #     true_pt = pd.read_csv(f"{save_dir}/{dataset}_true_pt.csv")
        # # else:
        # #     true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
        # #     true_pt.to_csv(f"{save_dir}/{dataset}_true_pt.csv")
        # #
        # # if os.path.exists(f"{save_dir}/{dataset}_expected_pt.csv"):
        # #     expected_pt = pd.read_csv(f"{save_dir}/{dataset}_expected_pt.csv")
        # # else:
        # #     expected_pt, expected_pt_solution = get_expected_pareto_front(df_pre_accuracy, df_true_accuracy, df_cost, model_list).run()
        # #     expected_pt.to_csv(f"{save_dir}/{dataset}_expected_pt.csv")
        #
        # # igs_res, igs_solution = iterative_greedy_search_(job_data, df_cost, df_pre_accuracy, model_list).run()
        #
        #
        # fig = plt.figure(figsize=(12, 7))
        # font = {  # 'family': 'serif',
        #     'color': 'black',
        #     'weight': 'normal',
        #     'size': 15,
        # }
        # for key, value in single_model_res.items():
        #     plt.scatter(value[0], value[1], alpha=1, label=key)
        #
        # # plt.scatter(true_pt['cost'], true_pt["true_accuracy"], alpha=1, c="yellow",
        # #             label='True PT')
        # #
        # # plt.scatter(expected_pt['cost'], expected_pt["true_accuracy"], alpha=1, c="firebrick",
        # #             label='expected PT')
        # # plt.scatter(expected_pt['cost'], expected_pt["expected_accuracy"], alpha=1, c="blue",
        # #             label='true result of expected PT')
        # plt.scatter(igs_res['cost'], igs_res["true_accuracy"], alpha=1, marker="^", c="blue",
        #             label='IGAP')
        # plt.xlabel('Cost (USD)', fontdict=font)
        # plt.ylabel('Accuracy', fontdict=font)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.grid(zorder=0, linestyle='--', axis='y')
        # plt.legend()
        # # lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.47, -0.13), ncol=7, fontsize=12)
        # plt.tight_layout()
        # plt.show()
        # # fig.savefig(f"{save_dir}/Comparison_{dataset}.png", dpi=500,
        # #             bbox_extra_artists=(lgd,), bbox_inches='tight')


