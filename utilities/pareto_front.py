from igs import *
from utilities import *


def get_cost_(job_data, model_data, levels_token, pareto_solutions):
    gpt4_input_price = model_data.loc[model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
    gpt_35_input_price = model_data.loc[model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]
    cost_list = []
    for solution in np.array(pareto_solutions):
        s_cost = np.zeros(len(solution))
        for i in range(len(solution)):
            token_num = job_data['TokenNum'][i]
            if solution[i] <= 2:
                s_cost[i] = (token_num + levels_token[solution[i]]) * gpt_35_input_price
            else:
                s_cost[i] = (token_num + levels_token[solution[i]]) * gpt4_input_price

        s_total_cost = np.sum(s_cost) / 1000
        cost_list.append(s_total_cost)
    return cost_list

def get_cost_list_(job_data, model_data):
    gpt4_input_price = model_data.loc[model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
    gpt_35_input_price = model_data.loc[model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]
    cost_list = []
    corresponding_index_list = []
    for i in range(len(job_data)):
        token_num = job_data['TokenNum'][i]
        if job_data['res_gpt3_simple'][i] == 1:
            corresponding_index_list.append(0)
            cost_list.append(float('inf'))
        elif job_data['res_gpt3_simple'][i] == 0:
            if job_data['res_gpt3_standard'][i] == 1:
                corresponding_index_list.append(1)
                cost_list.append(token_num * gpt_35_input_price)
            elif job_data['res_gpt3_standard'][i] == 0:
                if job_data['res_gpt3_enhance'][i] == 1:
                    corresponding_index_list.append(2)
                    cost_list.append(token_num * gpt_35_input_price)
                elif job_data['res_gpt3_enhance'][i] == 0:
                    if job_data['res_gpt4_simple'][i] == 1:
                        corresponding_index_list.append(3)
                        cost_list.append(token_num * gpt4_input_price)
                    elif job_data['res_gpt4_simple'][i] == 0:
                        if job_data['res_gpt4_standard'][i] == 1:
                            corresponding_index_list.append(4)
                            cost_list.append(token_num * gpt4_input_price)
                        elif job_data['res_gpt4_standard'][i] == 0:
                            if job_data['res_gpt4_enhance'][i] == 1:
                                corresponding_index_list.append(5)
                                cost_list.append(token_num * gpt4_input_price)
                            elif job_data['res_gpt4_enhance'][i] == 0:
                                corresponding_index_list.append(0)
                                cost_list.append(float('inf'))
    return cost_list, corresponding_index_list


def true_fitness_function_(job_data, model_data, levels_token, s_allocation):
    s_cost = np.zeros(len(s_allocation))
    gpt4_input_price = model_data.loc[model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
    gpt_35_input_price = model_data.loc[model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]

    for i in range(len(s_allocation)):
        token_num = job_data['TokenNum'][i]
        if s_allocation[i] <= 2:
            s_cost[i] = (token_num + levels_token[s_allocation[i]]) * gpt_35_input_price
        else:
            s_cost[i] = (token_num + levels_token[s_allocation[i]]) * gpt4_input_price

    res = 0
    for i in range(len(s_allocation)):
        if s_allocation[i] == 0:
            res += job_data['res_gpt3_simple'][i]
        elif s_allocation[i] == 1:
            res += job_data['res_gpt3_standard'][i]
        elif s_allocation[i] == 2:
            res += job_data['res_gpt3_enhance'][i]
        elif s_allocation[i] == 3:
            res += job_data['res_gpt4_simple'][i]
        elif s_allocation[i] == 4:
            res += job_data['res_gpt4_standard'][i]
        elif s_allocation[i] == 5:
            res += job_data['res_gpt4_enhance'][i]

    s_total_cost = np.sum(s_cost) / 1000
    s_accuracy_mean = res / len(job_data)
    # print(s_total_cost, s_accuracy_mean)
    return [s_total_cost, s_accuracy_mean]


def get_true_pareto_front_(job_data, model_data, levels_token):
    cost_list, corresponding_index_list = get_cost_list_(job_data, model_data)
    s_cheapest = [0] * len(job_data)
    true_pareto_solution = [s_cheapest]
    true_pareto = [true_fitness_function_(job_data, model_data, levels_token, s_cheapest)]
    new_solution = s_cheapest.copy()
    cost_list_copy = cost_list.copy()
    # check if there are other values except inf in cost list
    while len([cost for cost in cost_list_copy if cost != float('inf')]) > 0:
        # find the smallest value (exclude 0) in the cost list
        min_cost = min([cost for cost in cost_list_copy])
        # find the index of the smallest value in the cost list
        min_cost_index = cost_list_copy.index(min_cost)
        new_solution[min_cost_index] = corresponding_index_list[min_cost_index]
        true_pareto_solution.append(new_solution.copy())
        true_pareto.append(true_fitness_function_(job_data, model_data, levels_token, new_solution.copy()))
        # set the cost of the smallest value to inf
        cost_list_copy[min_cost_index] = float('inf')
    return true_pareto, true_pareto_solution

def get_true_nondominated_res_(pd_res):
    true_df = pd.DataFrame({'cost': pd_res['cost'], 'true_accuracy': pd_res['true_accuracy']})
    true_df = np.array(true_df)

    pareto_efficient_mask = is_pareto_(costs=true_df.copy())
    true_df = true_df[pareto_efficient_mask]
    true_df = pd.DataFrame(true_df, columns=['cost', 'true_accuracy']).sort_values(by=['cost'])

    return true_df