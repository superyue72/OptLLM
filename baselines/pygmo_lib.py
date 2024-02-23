import time
import pygmo as pg
from utilities import *

class PygmoAllocationProblem:
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])

    # Define objectives
    def fitness(self, x):
        s_cost = np.zeros(len(x))
        s_accuracy = np.zeros(len(x))
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
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
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.termination = termination

    def get_true_accuracy_obj_(self, pareto_solutions):
        parsed_num_list = []
        true_accuracy = []
        for solution in np.array(pareto_solutions):
            res = 0
            for i in range(len(solution)):
                res += self.df_true_accuracy.iloc[i, solution[i]]
            parsed_num_list.append(res)
            true_accuracy.append(res / len(pareto_solutions[0]))
        return true_accuracy

    def run(self):
        start_time = time.time()
        prob = pg.problem(
            PygmoAllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list))
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
        nondominated_res['true_accuracy'] = self.get_true_accuracy_obj_(nondominated_solutions)
        nondominated_res['time'] = elapsed_time
        print("MOPSO finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions


class MOEAD(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.termination = termination

    def get_true_accuracy_obj_(self, pareto_solutions):
            parsed_num_list = []
            true_accuracy = []
            for solution in np.array(pareto_solutions):
                res = 0
                for i in range(len(solution)):
                    res += self.df_true_accuracy.iloc[i, solution[i]]
                parsed_num_list.append(res)
                true_accuracy.append(res / len(pareto_solutions[0]))
            return true_accuracy

    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        prob = pg.problem(
            PygmoAllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list))
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
        nondominated_res['true_accuracy'] = self.get_true_accuracy_obj_(nondominated_solutions)
        nondominated_res['time'] = elapsed_time
        print("MOEAD finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions


class MOEADGEN(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.termination = termination

    def get_true_accuracy_obj_(self, pareto_solutions):
            parsed_num_list = []
            true_accuracy = []
            for solution in np.array(pareto_solutions):
                res = 0
                for i in range(len(solution)):
                    res += self.df_true_accuracy.iloc[i, solution[i]]
                parsed_num_list.append(res)
                true_accuracy.append(res / len(pareto_solutions[0]))
            return true_accuracy
    def run(self):
        start_time = time.time()
        levels_token = get_level_tokens_num_()
        prob = pg.problem(
            PygmoAllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list))
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
        nondominated_res['true_accuracy'] = self.get_true_accuracy_obj_(nondominated_solutions)
        nondominated_res['time'] = elapsed_time

        print("MOEADGEN finished and the searchiing time is: ", elapsed_time)
        return nondominated_res, nondominated_solutions






