import time
import pygmo as pg
from utilities import *

class PygmoAllocationProblem:
    def __init__(self, job_data, model_data, df_accuracy, levels_token, n_var, m_max=5):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.levels_token = levels_token
        self.n_var = n_var
        self.m_max = m_max

    # Define objectives
    def fitness(self, x):
        s_cost = np.zeros(len(x))
        s_accuracy = np.zeros(len(x))
        gpt4_input_price = self.model_data.loc[self.model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
        gpt_35_input_price = \
        self.model_data.loc[self.model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]

        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            token_num = self.job_data.TokenNum.iloc[i]
            # system = self.job_data['System'][i]
            # system_accuracy = self.df_accuracy.loc[self.df_accuracy['System'] == system].values[0, allocation + 1]

            if allocation <= 2:
                s_cost[i] = (token_num + self.levels_token[allocation]) * gpt_35_input_price
            else:
                s_cost[i] = (token_num + self.levels_token[allocation]) * gpt4_input_price

            s_accuracy[i] = self.df_accuracy.iloc[i, allocation]

        s_total_cost = np.sum(s_cost) / 1000
        s_accuracy_mean = np.mean(s_accuracy)
        # print(s_total_cost, s_accuracy_mean)
        return [s_total_cost, -s_accuracy_mean]

    # Return number of objectives
    def get_nobj(self):
        return 2

    # def get_nf(self):
    #     return 2

    def get_nix(self):
        return self.n_var

    # Return bounds of decision variables
    def get_bounds(self):
        return ([0] * self.n_var, [self.m_max] * self.n_var)

    # Return function name
    def get_name(self):
        return "Schaffer function N.1"


class MOPSO(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination
        self.m_max = 5

    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        prob = pg.problem(
            PygmoAllocationProblem(self.job_data, self.model_data, self.df_accuracy, levels_token, n_var=len(self.job_data), m_max=self.m_max))
        pop = pg.population(prob, size=100)
        algo = pg.algorithm(pg.nspso(gen=self.termination, omega=0.9375, c1=0.7631, c2=0.6387, v_coeff=0.1550))
        print("Start MOPSO searching!")
        pop = algo.evolve(pop=pop)
        elapsed_time = time.time() - start_time
        fits, vectors = pop.get_f(), pop.get_x()

        nondominated_solutions = pd.DataFrame(vectors).astype(int)
        nondominated_res = pd.DataFrame()
        nondominated_res['cost'] = fits[:, 0]
        nondominated_res['expected_accuracy'] = fits[:, 1] * (-1)

        pareto_efficient_mask = is_pareto_(costs=nondominated_res.copy())
        nondominated_res = nondominated_res[pareto_efficient_mask]
        nondominated_solutions = nondominated_solutions[pareto_efficient_mask]
        nondominated_res['true_accuracy'] = get_true_accuracy_(self.job_data, nondominated_solutions)
        nondominated_res['time'] = elapsed_time
        print("MOPSO finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions


class MOEAD(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination
        self.m_max = 5

    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        prob = pg.problem(
            PygmoAllocationProblem(self.job_data, self.model_data, self.df_accuracy, levels_token, n_var=len(self.job_data), m_max=self.m_max))
        pop = pg.population(prob, size=100)
        algo = pg.algorithm(pg.moead(gen=self.termination, weight_generation='grid', decomposition='weighted', neighbours=29))
        print("Start MOEAD searching!")
        pop = algo.evolve(pop=pop)
        elapsed_time = time.time() - start_time
        fits, vectors = pop.get_f(), pop.get_x()

        nondominated_solutions = pd.DataFrame(vectors).astype(int)
        nondominated_res = pd.DataFrame()
        nondominated_res['cost'] = fits[:, 0]
        nondominated_res['expected_accuracy'] = fits[:, 1] * (-1)

        pareto_efficient_mask = is_pareto_(costs=nondominated_res.copy())
        nondominated_res = nondominated_res[pareto_efficient_mask]
        nondominated_solutions = nondominated_solutions[pareto_efficient_mask]
        nondominated_res['true_accuracy'] = get_true_accuracy_(self.job_data, nondominated_solutions)
        nondominated_res['time'] = elapsed_time
        print("MOEAD finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions


class MOEADGEN(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination
        self.m_max = 5

    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        prob = pg.problem(
            PygmoAllocationProblem(self.job_data, self.model_data, self.df_accuracy, levels_token, n_var=len(self.job_data), m_max=self.m_max))
        pop = pg.population(prob, size=100)
        algo = pg.algorithm(pg.moead_gen(gen=self.termination, weight_generation='random', decomposition='weighted',
                                         neighbours=11))
        print("Start MOEADGEN searching!")
        pop = algo.evolve(pop=pop)
        elapsed_time = time.time() - start_time
        fits, vectors = pop.get_f(), pop.get_x()

        nondominated_solutions = pd.DataFrame(vectors).astype(int)
        nondominated_res = pd.DataFrame()
        nondominated_res['cost'] = fits[:, 0]
        nondominated_res['expected_accuracy'] = fits[:, 1] * (-1)

        pareto_efficient_mask = is_pareto_(costs=nondominated_res.copy())
        nondominated_res = nondominated_res[pareto_efficient_mask]
        nondominated_solutions = nondominated_solutions[pareto_efficient_mask]
        nondominated_res['true_accuracy'] = get_true_accuracy_(self.job_data, nondominated_solutions)
        nondominated_res['time'] = elapsed_time

        print("MOEADGEN finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions


if __name__ == '__main__':
    data_dir = "../data/log_parsing/mix"
    save_dir = "../output/log_parsing/mix_res"
    job_num_list = [1000, 3000, 5000, 7000]

    model_data = pd.read_csv(f"../data/model_data.csv")
    df_all_job = pd.read_csv(f"../data/log_parsing/check_table.csv")
    df_accuracy = pd.read_csv(f"../data/log_parsing/accuracy/sampling_accuracy_1.csv")
    levels_token = get_level_tokens_num_()

    for job_num in job_num_list:
        for i in range(1):
            start_time = time.time()
            job_data = pd.read_csv(f"{data_dir}/log_parsing_{job_num}_{i}.csv")
            job_data.index = range(job_data.shape[0])

            prob = pg.problem(PygmoAllocationProblem(job_data, model_data, df_accuracy,levels_token, n_var=len(job_data), m_max=5))
            pop = pg.population(prob, size=100)
            # algo = pg.algorithm(pg.maco(gen=100))
            algo = pg.algorithm(pg.nspso(gen=100))
            pop = algo.evolve(pop=pop)


            elapsed_time = time.time() - start_time

            fits, vectors = pop.get_f(), pop.get_x()
            # get the nondominated results from fits

            nondominated_solutions = pd.DataFrame(vectors).astype(int)
            nondominated_res = pd.DataFrame()
            nondominated_res['cost'] = fits[:, 0]
            nondominated_res['expected_accuracy'] = fits[:, 1] * (-1)

            pareto_efficient_mask = is_pareto_(costs=nondominated_res.copy())
            nondominated_res = nondominated_res[pareto_efficient_mask]
            nondominated_solutions = nondominated_solutions[pareto_efficient_mask]
            nondominated_res['true_accuracy'] = get_true_accuracy_(job_data, nondominated_solutions)
            nondominated_res['time'] = elapsed_time

            nondominated_res.to_csv(f"{save_dir}/nspso_res_{job_num}_{i}.csv", index=False)
            nondominated_solutions.to_csv(f"{save_dir}/nspso_solutions_{job_num}_{i}.csv", index=False)

            # plot the figure
            plt.scatter(nondominated_res['cost'], nondominated_res['expected_accuracy'])
            plt.xlabel('cost')
            plt.ylabel('expected_accuracy')
            plt.show()



            # s_cheapest, s_most_expensive, s_high_accuracy = local_search_(job_data, model_data, df_accuracy, sampling_sum=5, initial_solution_size = 50).get_exreme_solution_()
            # s1 = local_search_(job_data, model_data, df_accuracy, sampling_sum=5, initial_solution_size = 50).ls_fitness_function_(levels_token, s_cheapest)
            # s2 = local_search_(job_data, model_data, df_accuracy, sampling_sum=5, initial_solution_size = 50).ls_fitness_function_(levels_token, s_high_accuracy)
            # ref_point = [s1[1], s2[0]]
            # hv = pg.hypervolume(pop)
            # hv.compute(ref_point)
            # print(hv.compute(ref_point))



