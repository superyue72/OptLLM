import os
import pandas as pd
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV


def get_cost_(job_data, model_list):
    cost_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}_cost'
        cost_data[model] = job_data[cost_column]
    return cost_data

def get_accuracy_(job_data, model_list):
    accuracy_data = pd.DataFrame()
    for model in model_list:
        cost_column = f'{model}'
        accuracy_data[model] = job_data[cost_column]
    return accuracy_data

def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":  # note: future models may deviate from this
        num_tokens = len(encoding.encode(messages)) + 2
        return num_tokens
    elif model == "gpt-4" or model == "gpt-4-32k":
        num_tokens = len(encoding.encode(messages)) + 2
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. 
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def get_level_tokens_num_():
    language_models = ["gpt-3.5-turbo", "gpt-4"]
    levels_token = []
    prompt_templates = [
        "You will be provided with a log message delimited by backticks. Please extract the log template from this log message: ",
        "You will be provided with a log message delimited by backticks. You must abstract variables with ‘{placeholders}’ to extract the corresponding template. Print the input log’s template delimited by backticks. Log message: ",
        "You will be provided with a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with ‘{placeholders}‘ and output a static log template. Print the input log’s template delimited by backticks. Log message:",
    ]
    for model in language_models:
        for pt in prompt_templates:
            levels_token.append(num_tokens_from_messages(pt, model))

    return levels_token

def is_pareto_(costs, return_mask=True):
    costs = np.array(costs)
    costs[:, 1] = costs[:, 1] * -1
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def get_true_accuracy_obj_(df_true_accuracy, pareto_solutions):
    parsed_num_list = []
    true_accuracy = []
    for solution in np.array(pareto_solutions):
        res = 0
        for i in range(len(solution)):
            res += df_true_accuracy.iloc[i, solution[i]]
        parsed_num_list.append(res)
        true_accuracy.append(res/len(pareto_solutions[0]))
    return true_accuracy

def create_dir(dirname):
    path = os.getcwd() + '/' + dirname
    folder = os.path.exists(path)

    try:
        if not folder:
            os.makedirs(path, exist_ok=True)
    except OSError as err:
        print(err)

    return path + "/"



