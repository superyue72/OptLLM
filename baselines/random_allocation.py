import time
import random
import multiprocessing as mp
from utilities import *

class Random(object):
    def __init__(self, job_data, model_data, df_accuracy):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.initial_solution_size = 100
        self.scope = 5


    def initialization_random_(self):
        initial_solution = []
        for _ in range(self.initial_solution_size):
            num = random.randint(1, self.scope)
            candidate = list(range(0, self.scope))
            candidate_ID = random.sample(candidate, num)
            random_solution = []
            for j in range(len(self.job_data)):
                random_solution.append(random.choice(candidate_ID))
            initial_solution.append(random_solution)

        return initial_solution

    def ls_fitness_function_(self, levels_token, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))

        gpt4_input_price = self.model_data.loc[self.model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
        gpt_35_input_price = \
        self.model_data.loc[self.model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]

        for i, allocation in enumerate(s_allocation):
            token_num = self.job_data.TokenNum.iloc[i]
            if allocation <= 2:
                s_cost[i] = (token_num + levels_token[allocation]) * gpt_35_input_price
            else:
                s_cost[i] = (token_num + levels_token[allocation]) * gpt4_input_price

            s_accuracy[i] = self.df_accuracy.iloc[i, allocation]

        s_total_cost = np.sum(s_cost) / 1000
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        random_solution = self.initialization_random_()
        solution_res = []
        for solution in random_solution:
            solution_res.append(self.ls_fitness_function_(levels_token, solution))
        elapsed_time = time.time() - start_time

        solution_res = pd.DataFrame(solution_res, columns=['cost', 'expected_accuracy'])
        solution_res['time'] = elapsed_time
        solution_res['true_accuracy'] = get_true_accuracy_(self.job_data, random_solution)

        return solution_res, random_solution