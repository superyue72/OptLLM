import time

import pandas as pd
import numpy as np

class get_true_pareto_front(object):
    def __init__(self, df_acc, df_true_accuracy, df_cost, model_list):
        self.df_acc = df_acc
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list

    def get_cost_list_(self):
        cost_list = []
        corresponding_index_list = []
        for i in range(len(self.df_true_accuracy)):
            acc = self.df_true_accuracy.iloc[i]
            cost = self.df_cost.iloc[i].sort_values()
            if acc.sum() == 0:
                cost_list.append(float('inf'))
                corresponding_index_list.append(cost.index[0])
            else:
                for j in range(len(cost.index)):
                    if acc[cost.index[j]] == 1:
                        if j == 0:
                            cost_list.append(float('inf'))
                            corresponding_index_list.append(cost.index[0])
                            break
                        elif j == len(cost.index) - 1:
                            cost_list.append(float('inf'))
                            corresponding_index_list.append(cost.index[j])
                            break
                        else:
                            cost_list.append(cost[cost.index[j]])
                            corresponding_index_list.append(cost.index[j])
                            break
        index_list = []
        for index in corresponding_index_list:
            for j in range(len(self.model_list)):
                if index == self.model_list[j]:
                    index_list.append(j)
                    break

        return cost_list, index_list

    def get_low_cost_(self):
        s_low_cost = []
        for x in range(len(self.df_true_accuracy)):
            cost = self.df_cost.iloc[x]
            min_index = cost[cost == cost.min()].index[0]
            min_index_list = cost[cost == cost.min()].index.tolist()
            if len(min_index_list) > 1:
                accuracy = self.df_true_accuracy.iloc[x]
                all_accuracy = accuracy[min_index_list]
                max_accuracy = max(all_accuracy)
                min_index = all_accuracy[all_accuracy == max_accuracy].index[0]
            for j in range(len(self.model_list)):
                if min_index == self.model_list[j]:
                    s_low_cost.append(j)
        return s_low_cost

    def ls_fitness_function_(self, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_true_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

    def get_true_accuracy_obj_(self, pareto_solutions):
        parsed_num_list = []
        true_accuracy = []
        for solution in np.array(pareto_solutions):
            res = 0
            for i in range(len(solution)):
                res += self.df_acc.iloc[i, solution[i]]
            parsed_num_list.append(res)
            true_accuracy.append(res / len(solution))
        return true_accuracy
    def run(self):
        start_time = time.time()
        print('----------start getting the true pareto front----------')
        cost_list, corresponding_index_list = self.get_cost_list_()
        s_cheapest = self.get_low_cost_()

        true_pareto_solution = [s_cheapest]

        true_pareto = [self.ls_fitness_function_(s_cheapest)]
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
            true_pareto.append(self.ls_fitness_function_(new_solution.copy()))
            # set the cost of the smallest value to inf
            cost_list_copy[min_cost_index] = float('inf')

        df_true_pareto = pd.DataFrame(true_pareto, columns=['cost', 'true_accuracy'])
        elapsed_time = time.time() - start_time
        df_true_pareto['time'] = elapsed_time
        print('----------finish getting the true pareto front----------')

        return df_true_pareto, true_pareto_solution


