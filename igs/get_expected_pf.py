import pandas as pd
import numpy as np
import time
from tqdm import tqdm
class get_expected_pareto_front(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list):
        self.df_cost = df_cost
        self.model_list = model_list
        self.df_pred_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy

    def get_low_cost_(self):
        s_low_cost = []
        for x in range(len(self.df_cost)):
            cost = self.df_cost.iloc[x]
            min_index = cost[cost == cost.min()].index[0]
            min_index_list = cost[cost == cost.min()].index.tolist()
            if len(min_index_list) > 1:
                accuracy = self.df_pred_accuracy.iloc[x]
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
            s_accuracy[i] = self.df_pred_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

    def get_maximum_improvement_machine_(self, current_allocation, job_id):
        if job_id < 0 or job_id > len(self.df_cost) - 1:
            raise NotImplementedError(f"""the number of job exceeds the limit, please check the job_id: {job_id}""")

        accuracy = self.df_pred_accuracy.iloc[job_id]
        cost_list = self.df_cost.iloc[job_id]

        accuracy_diff = []
        cost_list_diff = []
        job_index = []
        for ele in range(len(cost_list)):
            if current_allocation[job_id] < 0 or current_allocation[job_id] > len(self.df_cost) - 1:
                print(current_allocation[job_id])
                raise NotImplementedError(
                    f"""the allocation exceeds the limit, please check the job_id: {current_allocation[job_id]}""")
            if ele != current_allocation[job_id]:
                cost_list_diff.append(cost_list.iloc[ele] - cost_list.iloc[current_allocation[job_id]])
                accuracy_diff.append(accuracy.iloc[ele] - accuracy.iloc[current_allocation[job_id]])
                job_index.append(ele)

        improvement_list = []
        improvement_index = []
        for i in range(len(cost_list_diff)):
            if accuracy_diff[i] <= 0:
                improvement_list.append(0)
                improvement_index.append(job_index[i])
            else:
                improvement_list.append(accuracy_diff[i] / cost_list_diff[i])
                improvement_index.append(job_index[i])
        return improvement_index[improvement_list.index(max(improvement_list))], max(improvement_list)

    def maximum_improvement_per_job_(self, current_allocation):
        all_improvement = []
        all_level = []
        for i in range(len(current_allocation)):
            improvement_index, max_improvement = self.get_maximum_improvement_machine_(current_allocation, i)
            all_improvement.append(max_improvement)
            all_level.append(improvement_index)
        return all_improvement, all_level

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

    def run(self):
        start_time = time.time()
        print('----------Start to get expected pareto front!----------')
        s_cheapest = self.get_low_cost_()
        new_solution = s_cheapest.copy()
        res_from_cheapest = []
        all_solutions_from_cheapest = []
        all_solutions_from_cheapest.append(new_solution.copy())
        res_from_cheapest.append(self.ls_fitness_function_(new_solution.copy()))
        all_improvement, all_level = self.maximum_improvement_per_job_(new_solution)
        max_improvement = max(all_improvement)
        while max_improvement > 0:
            max_improvement_index = all_improvement.index(max(all_improvement))
            max_level = all_level[max_improvement_index]
            new_solution[max_improvement_index] = max_level
            all_solutions_from_cheapest.append(new_solution.copy())
            res_from_cheapest.append(self.ls_fitness_function_(new_solution.copy()))
            new_index, new_improvement = self.get_maximum_improvement_machine_(new_solution, max_improvement_index)
            all_improvement[max_improvement_index] = new_improvement
            all_level[max_improvement_index] = new_index
            max_improvement = max(all_improvement)

        print(f"--- {time.time() - start_time} seconds ---")

        res_from_cheapest = pd.DataFrame(res_from_cheapest, columns=['cost', 'expected_accuracy'])
        res_from_cheapest['true_accuracy'] = self.get_true_accuracy_obj_(all_solutions_from_cheapest)

        return res_from_cheapest, all_solutions_from_cheapest

