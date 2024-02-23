import time
import multiprocessing
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.rnd import RandomSelection
from utilities import *


class AllocationProblem(ElementwiseProblem):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, num_var, m_max, **kwargs):
        super().__init__(n_var=num_var, n_obj=2, n_ieq_constr=0, xl=0, xu=m_max, vtype=int, **kwargs)
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.m_max = m_max

    def _evaluate(self, x, out, *args, **kwargs):
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
        out["F"] = [s_total_cost, -s_accuracy_mean]

class AllocationProblemWithReferPoint(ElementwiseProblem):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, num_var, m_max, **kwargs):
        super().__init__(n_var=num_var, n_obj=2, n_ieq_constr=0, xl=0, xu=m_max, vtype=int, **kwargs)
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.m_max = m_max

    def _evaluate(self, x, out, *args, **kwargs):
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
        out["F"] = [s_total_cost, -s_accuracy_mean]

class sms_emoa(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
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
        MyPorblem = AllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                      self.n_var, self.m_max)
        # termination = get_termination("time", self.termination)
        termination = get_termination("n_gen", self.termination)
        algorithm = SMSEMOA(pop_size=100,
                            crossover=SBX(prob=0.9167, eta=29),
                            mutation=PM(prob=0.7388, eta=16),
                            sampling=FloatRandomSampling(),
                            )
        print("Start SMS-EMOA searching!")
        res = minimize(MyPorblem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)
        smsemoa_res = res.F
        smsemoa_solution = res.X
        smsemoa_solutions_int = smsemoa_solution.astype(int)
        smsemoa_res[:, 1] = smsemoa_res[:, 1] * -1
        elapsed_time = time.time() - start_time

        smsemoa_res = pd.DataFrame(smsemoa_res, columns=['cost', 'expected_accuracy'])
        smsemoa_res['time'] = elapsed_time
        smsemoa_res['true_accuracy'] = self.get_true_accuracy_obj_( smsemoa_solutions_int)
        print("SMS-EMOA finished and the searchiing time is: ", elapsed_time)

        return smsemoa_res, smsemoa_solutions_int #, elapsed_time

class nsga2(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
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
        print("n_var: ", self.n_var)
        print("m_max: ", self.m_max)
        MyPorblem = AllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                      self.n_var, self.m_max)
        # termination = get_termination("time", self.termination)
        termination = get_termination("n_gen", self.termination)
        algorithm = NSGA2(pop_size=100,
                          sampling=LHS(),
                          selection=RandomSelection(),
                          crossover=SBX(prob=0.9053, eta=24, vtype=int, repair=RoundingRepair()),
                          mutation=PM(prob=0.0023, eta=6, vtype=int, repair=RoundingRepair()),
                          eliminate_duplicates=True,
                          )
        print("Start NSGA-2 searching!")
        res_nsga = minimize(MyPorblem,
                            algorithm,
                            termination,
                            seed=1,
                            verbose=False)
        nsga_res = res_nsga.F
        nsga_solution = res_nsga.X
        nsga_res[:, 1] = nsga_res[:, 1] * -1
        elapsed_time = time.time() - start_time

        nsga2_res = pd.DataFrame(nsga_res, columns=['cost', 'expected_accuracy'])
        nsga2_res['time'] = elapsed_time
        nsga2_res['true_accuracy'] = self.get_true_accuracy_obj_(nsga_solution)
        print("NSGA-2 finished and the searchiing time is: ", elapsed_time)
        # return nsga_res, nsga_solution, elapsed_time
        return nsga2_res, nsga_solution
class rnsga2(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination

    def get_exreme_solution_(self):
        def get_high_accuracy_():
            s_high_accuracy = []
            for x in range(len(self.df_pre_accuracy)):
                accuracy = self.df_pre_accuracy.iloc[x]
                max_index = accuracy[accuracy == accuracy.max()].index[0]
                max_index_list = accuracy[accuracy == accuracy.max()].index.tolist()
                if len(max_index_list) > 1:
                    # print(max_index_list)
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


    def fitness_function_(self, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        return [s_total_cost, s_accuracy_mean]

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
        # termination = get_termination("time", self.termination)
        termination = get_termination("n_gen", self.termination)
        s_cheapest, s_high_accuracy = self.get_exreme_solution_()
        s1_obj1, s1_obj2 = self.fitness_function_(s_cheapest)
        s2_obj1, s2_obj2 = self.fitness_function_(s_high_accuracy)
        ref_points = np.array([[s1_obj1, -s1_obj2], [s2_obj1, -s2_obj2]])
        MyPorblem = AllocationProblemWithReferPoint(self.df_pre_accuracy,self.df_true_accuracy,self.df_cost,
                                                    self.model_list,
                                                    num_var=len(self.df_true_accuracy),
                                                    m_max=len(self.df_true_accuracy.iloc[0]) - 1)
        algorithm = RNSGA2(
            ref_points=ref_points,
            pop_size=100,
            epsilon=0.1043,
            normalization='front',
            extreme_points_as_reference_points=True,
            weights=np.array([0.5, 0.5]))
        print("Start R-NSGA-2 searching!")
        res_rnsga2 = minimize(MyPorblem,
                       algorithm,
                       save_history=True,
                       termination=termination,
                       seed=1,
                       disp=False)

        rnsga2_res = res_rnsga2.F
        rnsga2_solution = res_rnsga2.X
        rnsga2_solutions_int = rnsga2_solution.astype(int)
        rnsga2_res[:, 1] = rnsga2_res[:, 1] * -1
        elapsed_time = time.time() - start_time

        r_nsga2_res = pd.DataFrame(rnsga2_res, columns=['cost', 'expected_accuracy'])
        r_nsga2_res['time'] = elapsed_time
        r_nsga2_res['true_accuracy'] = self.get_true_accuracy_obj_(rnsga2_solutions_int)
        print("R-NSGA-2 finished and the searchiing time is: ", elapsed_time)

        return r_nsga2_res, rnsga2_solutions_int