import time

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead

class GetCostProblem(Problem):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, min_acc, num_var, m_max, **kwargs):
        super().__init__(n_var=num_var, n_obj=1, n_ieq_constr=1, xl=0, xu=m_max, vtype=int, **kwargs)
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.m_max = m_max
        self.min_acc = min_acc

    def _evaluate(self, x, out, *args, **kwargs):
        s_cost = np.zeros(len(x))
        s_accuracy = np.zeros(len(x))
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)

        for j in range(len(s_allocation)):
            s = s_allocation[j]
            cost  = 0
            acc = 0
            for i, allocation in enumerate(s):
                cost = cost + self.df_cost.iloc[i, allocation]
                acc = acc + self.df_pre_accuracy.iloc[i, allocation]
            s_cost[j] = cost
            s_accuracy[j] = acc/len(s)
        out["F"] = [s_cost]
        out["G"] = [self.min_acc - s_accuracy]


class GetAccProblem(Problem):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, max_cost, num_var, m_max, **kwargs):
        super().__init__(n_var=num_var, n_obj=1, n_ieq_constr=1, xl=0, xu=m_max, vtype=int, **kwargs)
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.m_max = m_max
        self.max_cost = max_cost

    def _evaluate(self, x, out, *args, **kwargs):
        s_cost = np.zeros(len(x))
        s_accuracy = np.zeros(len(x))
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)

        for j in range(len(s_allocation)):
            s = s_allocation[j]
            cost  = 0
            acc = 0
            for i, allocation in enumerate(s):
                cost = cost + self.df_cost.iloc[i, allocation]
                acc = acc + self.df_pre_accuracy.iloc[i, allocation]
            s_cost[j] = cost
            s_accuracy[j] = acc/len(s)
            # print(s_accuracy[j])

        # for i, allocation in enumerate(s_allocation):
        #     s_cost[i] = self.df_cost.iloc[i, allocation]
        #     s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        # s_total_cost = np.sum(s_cost)
        # s_accuracy_mean = np.mean(s_accuracy)
        out["F"] = [-s_accuracy]
        out["G"] = [s_cost - self.max_cost]
        # out["F"] = [s_cost]
        # out["G"] = [self.min_acc-s_accuracy]


##################Maximize accuracy under cost constraint##################

class ga_for_acc(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.max_cost = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]
        print("cost is", cost)
        s_accuracy = acc / len(s_allocation)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetAccProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list, self.max_cost, self.n_var, self.m_max)
        start_time = time.time()
        algorithm = GA(
            pop_size=100,
            eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("ga expected acc is ", ga_res)
            print("ga true acc is", s_accuracy)
            print("ga takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time



class de_for_acc(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.max_cost = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]
        print("cost is", cost)
        s_accuracy = acc / len(s_allocation)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetAccProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list, self.max_cost, self.n_var, self.m_max)
        start_time = time.time()

        algorithm = DE(
            pop_size=100,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("de expected acc is ", ga_res)
            print("de true acc is", s_accuracy)
            print("de takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time


class pso_for_acc(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.max_cost = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]

        s_accuracy = acc / len(s_allocation)
        print("cost is", cost)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetAccProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list, self.max_cost, self.n_var, self.m_max)
        start_time = time.time()

        algorithm = PSO()

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("pso expected acc is ", ga_res)
            print("pso true acc is", s_accuracy)
            print("pso takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time


class nm_for_acc(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.max_cost = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]

        s_accuracy = acc / len(s_allocation)
        print("cost is", cost)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetAccProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                self.max_cost, self.n_var, self.m_max)
        start_time = time.time()

        algorithm = NelderMead()

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        s_accuracy = self.get_true_accuracy(ga_solution)
        print("expected acc is ", ga_res)
        print("true acc is", s_accuracy)
        print("it takes", elapsed_time)
        print(ga_res, elapsed_time)
        return ga_res, s_accuracy, ga_solution, elapsed_time


#######################Minimize cost under accuracy constraint######################
class ga_for_cost(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.const_acc = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]
        s_accuracy = acc / len(s_allocation)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetCostProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                 self.const_acc, self.n_var, self.m_max)
        start_time = time.time()
        algorithm = GA(
            pop_size=100,
            eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        print("ga cost is ", ga_res)
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("ga true acc is", s_accuracy)
            print("ga takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time


class de_for_cost(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.const_acc = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]
        s_accuracy = acc / len(s_allocation)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetCostProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                 self.const_acc, self.n_var, self.m_max)
        start_time = time.time()
        algorithm = DE(
            pop_size=100,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        print("de cost is ", ga_res)
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("de true acc is", s_accuracy)
            print("de takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time


class pso_for_cost(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, const_value, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.n_var = len(self.df_true_accuracy)
        self.m_max = len(self.df_true_accuracy.iloc[0])
        self.termination = termination
        self.const_acc = const_value

    def get_true_accuracy(self, x):
        cost = 0
        acc = 0
        s_allocation = x.copy()
        # transfer s_allocation to int
        s_allocation = s_allocation.astype(int)
        for i, allocation in enumerate(s_allocation):
            cost = cost + self.df_cost.iloc[i, allocation]
            acc = acc + self.df_true_accuracy.iloc[i, allocation]
        s_accuracy = acc / len(s_allocation)
        return s_accuracy

    def run(self):
        termination = get_termination("n_gen", self.termination)
        problem = GetCostProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list,
                                 self.const_acc, self.n_var, self.m_max)
        start_time = time.time()
        algorithm = PSO()

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=False)

        elapsed_time = time.time() - start_time
        ga_res = res.F
        ga_solution = res.X
        print("pso cost is ", ga_res)
        if ga_solution is None:
            print("Cannot find the available solution")
            return None, None, None, None
        else:
            s_accuracy = self.get_true_accuracy(ga_solution)
            print("pso true acc is", s_accuracy)
            print("pso takes", elapsed_time)
            return ga_res, s_accuracy, ga_solution, elapsed_time