import heapq
import multiprocessing as mp
import random
import time
from itertools import combinations
from utilities import *
from tqdm import tqdm
class iterative_greedy_search_(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.scope = len(self.df_true_accuracy.iloc[0])
        self.density = 100

    def is_pareto_(self, costs, return_mask=True):
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

    def ls_fitness_function_(self, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

    def get_exreme_solution_(self):
        def get_high_accuracy_():
            s_high_accuracy = []
            for x in range(len(self.df_pre_accuracy)):
                accuracy = self.df_pre_accuracy.iloc[x]
                max_index = accuracy[accuracy == accuracy.max()].index[0]
                max_index_list = accuracy[accuracy == accuracy.max()].index.tolist()
                if len(max_index_list) > 1:
                    print(max_index_list)
                    cost = self.df_cost.iloc[x]
                    all_cost = cost[max_index_list]
                    min_cost = min(all_cost)
                    max_index = all_cost[all_cost == min_cost].index[0]
                for j in range(len(self.model_list)):
                    if max_index == self.model_list[j]:
                        s_high_accuracy.append(j)
            return s_high_accuracy

        def get_low_cost_():
            s_low_cost = []
            for x in range(len(self.df_pre_accuracy)):
                cost = self.df_cost.iloc[x]
                min_index = cost[cost == cost.min()].index[0]
                min_index_list = cost[cost == cost.min()].index.tolist()
                if len(min_index_list) > 1:
                    accuracy = self.df_pre_accuracy.iloc[x]
                    all_accuracy = accuracy[min_index_list]
                    max_accuracy = max(all_accuracy)
                    min_index = all_accuracy[all_accuracy == max_accuracy].index[0]
                for j in range(len(self.model_list)):
                    if min_index == self.model_list[j]:
                        s_low_cost.append(j)
            return s_low_cost

        s_high_accuracy = get_high_accuracy_()
        s_cheapest = get_low_cost_()

        return s_cheapest, s_high_accuracy

    def single_job_score_table_generation(self, current_allocation, job_id):
        accuracy = self.df_pre_accuracy.iloc[job_id]
        cost_list = self.df_cost.iloc[job_id]

        accuracy_diff = []
        cost_list_diff = []
        job_index = []
        for ele in range(len(cost_list)):
            if ele != current_allocation[job_id]:
                cost_list_diff.append(cost_list.iloc[ele] - cost_list.iloc[current_allocation[job_id]])
                accuracy_diff.append(accuracy.iloc[ele] - accuracy.iloc[current_allocation[job_id]])
                job_index.append(ele)

        accuracy_change_unit_cost = []
        accuracy_change_unit_cost_index = []

        cost_diff_copy = cost_list_diff.copy()
        accuracy_diff_copy = accuracy_diff.copy()
        for i in range(len(cost_list_diff)):
            if cost_list_diff[i] <= 0 and accuracy_diff[i] >= 0 and (accuracy_diff[i] - cost_list_diff[i]) > 0:
                accuracy_change_unit_cost.append(float('inf'))
                accuracy_change_unit_cost_index.append(job_index[i])
            elif cost_list_diff[i] > 0 and accuracy_diff[i] > 0:
                accuracy_change_unit_cost.append(accuracy_diff[i] / cost_list_diff[i])  # bigger is better
                accuracy_change_unit_cost_index.append(job_index[i])
            elif cost_list_diff[i] < 0 and accuracy_diff[i] < 0:
                accuracy_change_unit_cost.append(-accuracy_diff[i] / cost_list_diff[i])  # bigger is worse
                accuracy_change_unit_cost_index.append(job_index[i])
            else:
                accuracy_change_unit_cost.append(float('-inf'))
                accuracy_change_unit_cost_index.append(job_index[i])
                cost_diff_copy[i] = float('-inf')
                accuracy_diff_copy[i] = float('-inf')
        return accuracy_diff_copy, cost_diff_copy, accuracy_change_unit_cost, accuracy_change_unit_cost_index

    def score_table_generation(self, current_allocation):
        accuracy_diff_list = []
        cost_list_diff_list = []
        accuracy_change_unit_cost_list = []
        accuracy_change_unit_cost_index_list = []
        for i in range(len(current_allocation)):
            accuracy_diff, cost_list_diff, accuracy_change_unit_cost, accuracy_change_unit_cost_index = (
                self.single_job_score_table_generation(current_allocation, i))
            accuracy_diff_list.append(accuracy_diff)
            cost_list_diff_list.append(cost_list_diff)
            accuracy_change_unit_cost_list.append(accuracy_change_unit_cost)
            accuracy_change_unit_cost_index_list.append(accuracy_change_unit_cost_index)

        return np.array(accuracy_diff_list), np.array(cost_list_diff_list), np.array(
            accuracy_change_unit_cost_list), np.array(accuracy_change_unit_cost_index_list)

    def destruction(self, cost_gap, solution):
        new_s = solution.copy()
        candidate_pair = {}
        np_accuracy_diff_list, np_cost_list_diff_list, np_accuracy_change_unit_cost_list, np_index_list = self.score_table_generation(new_s)
        for i in range(len(new_s)):
            cost_list = np_cost_list_diff_list[i]
            condition_cost = (cost_list != float('-inf')) & (cost_list < 0)
            if np.any(condition_cost):
                min_cost_saving_index = np.where(cost_list == min(cost_list[condition_cost]))
                index_y = min_cost_saving_index[0][0]
                candidate_pair[(i, np_index_list[i][index_y])] = min(
                    cost_list[condition_cost])
        candidate_pair = sorted(candidate_pair.items(), key=lambda x: x[1], reverse=False)

        cost_saving = 0
        for j in range(len(candidate_pair)):
            new_s[candidate_pair[j][0][0]] = candidate_pair[j][0][1]
            cost_saving -= candidate_pair[j][1]
            if cost_saving >= cost_gap:
                return new_s

    def optimise_to_nondominated_(self, current_solution):
        new_solution = current_solution.copy()
        np_accuracy_diff_list, np_cost_list_diff_list, np_accuracy_change_unit_cost_list, np_index_list = self.score_table_generation(
            new_solution)
        if np.any(np.isposinf(np_accuracy_change_unit_cost_list)):
            inf_index_row1, inf_index_col1 = np.where(np.isposinf(np_accuracy_change_unit_cost_list))
            new_inf_index = np.unique(inf_index_row1, return_index=True)
            # get the coressponding col index in inf_index_col
            inf_index_row = new_inf_index[0]
            inf_index_col = inf_index_col1[new_inf_index[1]]
            for i in range(len(inf_index_row)):
                new_solution[inf_index_row[i]] = np_index_list[inf_index_row[i]][
                    inf_index_col[i]]
                accuracy_diff, cost_list_diff, accuracy_change_unit_cost, accuracy_change_unit_cost_index = (
                    self.single_job_score_table_generation(new_solution, inf_index_row[i]))
                np_accuracy_change_unit_cost_list[inf_index_row[i]] = accuracy_change_unit_cost
                np_index_list[inf_index_row[i]] = accuracy_change_unit_cost_index

        condition_i = (np_accuracy_change_unit_cost_list != float('inf')) & (np_accuracy_change_unit_cost_list > 0)
        condition_d = (np_accuracy_change_unit_cost_list != float('-inf')) & (np_accuracy_change_unit_cost_list < 0)

        if np_accuracy_change_unit_cost_list[condition_d].size == 0 or np_accuracy_change_unit_cost_list[
            condition_i].size == 0:
            return new_solution.copy()
        max_increase_value = max(np_accuracy_change_unit_cost_list[condition_i])
        max_decrease_value = max(np_accuracy_change_unit_cost_list[condition_d])

        while abs(max_decrease_value) < max_increase_value and max_increase_value > 0 and abs(max_decrease_value) > 0:
            max_increase_value_index_row, max_increase_value_index_col = np.where(
                np_accuracy_change_unit_cost_list == max_increase_value)
            max_decrease_value_index_row, max_decrease_value_index_col = np.where(
                np_accuracy_change_unit_cost_list == max_decrease_value)
            for i in range(len(max_increase_value_index_row)):
                new_solution[max_increase_value_index_row[i]] = \
                    np_index_list[max_increase_value_index_row[i]][max_increase_value_index_col[i]]
                # job_id = max_increase_value_index[0][i]
                accuracy_diff, cost_list_diff, accuracy_change_unit_cost, accuracy_change_unit_cost_index = (
                    self.single_job_score_table_generation(new_solution, max_increase_value_index_row[i]))
                np_accuracy_change_unit_cost_list[max_increase_value_index_row[i]] = accuracy_change_unit_cost
                np_index_list[max_increase_value_index_row[i]] = accuracy_change_unit_cost_index
            for i in range(len(max_decrease_value_index_row)):
                new_solution[max_decrease_value_index_row[i]] = \
                    np_index_list[max_decrease_value_index_row[i]][
                        max_decrease_value_index_col[i]]
                accuracy_diff, cost_list_diff, accuracy_change_unit_cost, accuracy_change_unit_cost_index = (
                    self.single_job_score_table_generation(new_solution, max_decrease_value_index_row[i]))
                np_accuracy_change_unit_cost_list[max_decrease_value_index_row[i]] = accuracy_change_unit_cost
                np_index_list[max_decrease_value_index_row[i]] = accuracy_change_unit_cost_index

            while np.any(np.isposinf(np_accuracy_change_unit_cost_list)):
                inf_index_row1, inf_index_col1 = np.where(np.isposinf(np_accuracy_change_unit_cost_list))
                new_inf_index = np.unique(inf_index_row1, return_index=True)
                # get the coressponding col index in inf_index_col
                inf_index_row = new_inf_index[0]
                inf_index_col = inf_index_col1[new_inf_index[1]]
                for i in range(len(inf_index_row)):
                    new_solution[inf_index_row[i]] = np_index_list[inf_index_row[i]][
                        inf_index_col[i]]
                    accuracy_diff, cost_list_diff, accuracy_change_unit_cost, accuracy_change_unit_cost_index = self.single_job_score_table_generation(new_solution, inf_index_row[i])
                    np_accuracy_change_unit_cost_list[inf_index_row[i]] = accuracy_change_unit_cost
                    np_index_list[inf_index_row[i]] = accuracy_change_unit_cost_index
            max_increase_value = max(
                np_accuracy_change_unit_cost_list[np_accuracy_change_unit_cost_list != float('inf')])
            condition = (np_accuracy_change_unit_cost_list != float('-inf')) & (np_accuracy_change_unit_cost_list < 0)
            max_decrease_value = max(np_accuracy_change_unit_cost_list[condition])

        new_s = new_solution.copy()
        return new_s

    def get_true_accuracy_obj_(self, pareto_solutions):
        parsed_num_list = []
        true_accuracy = []
        for solution in tqdm(np.array(pareto_solutions)):
            res = 0
            for i in range(len(solution)):
                res += self.df_true_accuracy.iloc[i, solution[i]]
            parsed_num_list.append(res)
            true_accuracy.append(res / len(pareto_solutions[0]))
        return true_accuracy

    def get_single_model_results_(self, model):
        model_column = f'{model}'
        f_accuracy = self.df_true_accuracy[model_column].sum() / len(self.df_true_accuracy)
        f_cost = self.df_cost[model_column].sum()
        return [f_cost, f_accuracy]


    def run(self):
        start_time = time.time()
        print('----------Start to get expected pareto front!----------')
        s_cheapest, s_high_accuracy = self.get_exreme_solution_()
        res_cheapest = self.ls_fitness_function_(s_cheapest)
        res_high_accuracy = self.ls_fitness_function_(s_high_accuracy)
        ls_solutions = [s_cheapest, s_high_accuracy]
        nondominated_res = np.vstack([res_cheapest, res_high_accuracy])

        cost_diff = res_high_accuracy[0] - res_cheapest[0]
        acc_diff = res_high_accuracy[1] - res_cheapest[1]
        cost_grid = np.linspace(0, cost_diff, self.density)

        s = s_high_accuracy.copy()
        destruction_set = []
        reconstruction_set = []
        destruction_res_set = []
        reconstruction_res_set = []
        for i in range(1, self.density):
            print(i)
            s_destruction = self.destruction(cost_grid[i], s)
            if s_destruction is None:
                break
            s_reconstruction = self.optimise_to_nondominated_(s_destruction)
            s = s_reconstruction.copy()
            destruction_set.append(s_destruction.copy())
            reconstruction_set.append(s_reconstruction.copy())

        for i in range(len(destruction_set)):
            destruction_res_set.append(self.ls_fitness_function_(destruction_set[i]))
        for i in range(len(reconstruction_set)):
            reconstruction_res_set.append(self.ls_fitness_function_(reconstruction_set[i]))


        nondominated_res = np.vstack((nondominated_res, reconstruction_res_set))
        ls_solutions.extend(reconstruction_set)

        pareto_efficient_mask = self.is_pareto_(nondominated_res.copy())
        nondominated_solutions = np.array(ls_solutions)[pareto_efficient_mask]
        nondominated_res = nondominated_res[pareto_efficient_mask]

        elapsed_time = time.time() - start_time
        igs_res = pd.DataFrame(nondominated_res, columns=['cost', 'expected_accuracy'])

        igs_res['true_accuracy'] = self.get_true_accuracy_obj_(nondominated_solutions)
        print("Local search without initial and fill gap finished and the searchiing time is: ", elapsed_time)

        return igs_res, nondominated_solutions


