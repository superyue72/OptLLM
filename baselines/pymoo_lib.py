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
    def __init__(self, job_data, model_data, df_accuracy, n_var, m_max=5, **kwargs):
        super().__init__(n_var=len(job_data), n_obj=2, n_ieq_constr=0, xl=0, xu=m_max, vtype=int, **kwargs)
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.n_var = n_var
        self.m_max = m_max

    def num_tokens_from_messages(self, messages, model):
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

    def get_level_tokens_num_(self):
        language_models = ["gpt-3.5-turbo", "gpt-4"]
        levels_token = []
        prompt_templates = [
            "You will be provided with a log message delimited by backticks. Please extract the log template from this log message: ",
            "You will be provided with a log message delimited by backticks. You must abstract variables with ‘{placeholders}’ to extract the corresponding template. Print the input log’s template delimited by backticks. Log message: ",
            "You will be provided with a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with ‘{placeholders}‘ and output a static log template. Print the input log’s template delimited by backticks. Log message:",
        ]
        for model in language_models:
            for pt in prompt_templates:
                levels_token.append(self.num_tokens_from_messages(pt, model))

        return levels_token

    def _evaluate(self, x, out, *args, **kwargs):
        levels_token = self.get_level_tokens_num_()
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
                s_cost[i] = (token_num + levels_token[allocation]) * gpt_35_input_price
            else:
                s_cost[i] = (token_num + levels_token[allocation]) * gpt4_input_price

            s_accuracy[i] = self.df_accuracy.iloc[i, allocation]

        s_total_cost = np.sum(s_cost) / 1000
        s_accuracy_mean = np.mean(s_accuracy)
        # print(s_total_cost, s_accuracy_mean)
        out["F"] = [s_total_cost, -s_accuracy_mean]

class AllocationProblemWithReferPoint(ElementwiseProblem):
    def __init__(self, job_data, model_data, df_accuracy, levels_token, n_var, m_max=5, **kwargs):
        super().__init__(n_var=len(job_data), n_obj=2, n_ieq_constr=0, xl=0, xu=m_max, vtype=int, **kwargs)
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.levels_token = levels_token
        self.n_var = n_var
        self.m_max = m_max

    def _evaluate(self, x, out, *args, **kwargs):
        s_cost = np.zeros(len(x))
        s_accuracy = np.zeros(len(x))
        gpt4_input_price = self.model_data.loc[self.model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
        gpt_35_input_price = self.model_data.loc[self.model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]

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
        out["F"] = [s_total_cost, -s_accuracy_mean]

class SMS_EMOA(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination

    def run(self):
        start_time = time.time()
        # n_proccess = os.cpu_count()
        # pool = multiprocessing.Pool(n_proccess)
        # runner = StarmapParallelization(pool.starmap)
        # MyPorblem = AllocationProblem(job_data=self.job_data, model_data=self.model_data, df_accuracy=self.df_accuracy,
        #                                         n_var=len(self.job_data), m_max=5, elementwise_runner=runner)

        MyPorblem = AllocationProblem(job_data=self.job_data, model_data=self.model_data, df_accuracy=self.df_accuracy,
                                      n_var=len(self.job_data), m_max=5)
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
        smsemoa_res['true_accuracy'] = get_true_accuracy_(self.job_data, smsemoa_solutions_int)
        print("SMS-EMOA finished and the searchiing time is: ", elapsed_time)

        return smsemoa_res, smsemoa_solutions_int #, elapsed_time

class nsga2(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination

    def run(self):
        start_time = time.time()
        n_proccess = os.cpu_count()
        # pool = multiprocessing.Pool(n_proccess)
        # runner = StarmapParallelization(pool.starmap)
        # MyPorblem = AllocationProblem(job_data=self.job_data, model_data=self.model_data, df_accuracy=self.df_accuracy,
        #                                         n_var=len(self.job_data), m_max=5, elementwise_runner=runner)
        MyPorblem = AllocationProblem(job_data=self.job_data, model_data=self.model_data, df_accuracy=self.df_accuracy,
                                                n_var=len(self.job_data), m_max=5)
        # termination = get_termination("time", self.termination)
        termination = get_termination("n_gen", self.termination)
        algorithm = NSGA2(pop_size=100,
                          sampling=LHS(),
                          selection=TournamentSelection(),
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
        nsga2_res['true_accuracy'] = get_true_accuracy_(self.job_data, nsga_solution)
        print("NSGA-2 finished and the searchiing time is: ", elapsed_time)
        # return nsga_res, nsga_solution, elapsed_time
        return nsga2_res, nsga_solution
class r_nsga2(object):
    def __init__(self, job_data, model_data, df_accuracy, termination):
        self.job_data = job_data
        self.model_data = model_data
        self.df_accuracy = df_accuracy
        self.termination = termination
        self.scope = len(self.df_accuracy.iloc[0]) - 1  # minus the name column

    def get_exreme_solution_(self):
        def get_high_accuracy_():
            s_high_accuracy = []
            print(len(self.job_data))
            for x in range(len(self.job_data)):
                accuracy = self.df_accuracy.iloc[x]
                # return the max accuracy, if there are more than one, return the first one
                max_index = accuracy[accuracy == accuracy.max()].index[0]
                if max_index == 'res_gpt3_simple':
                    s_high_accuracy.append(0)
                elif max_index == 'res_gpt3_standard':
                    s_high_accuracy.append(1)
                elif max_index == 'res_gpt3_enhance':
                    s_high_accuracy.append(2)
                elif max_index == 'res_gpt4_simple':
                    s_high_accuracy.append(3)
                elif max_index == 'res_gpt4_standard':
                    s_high_accuracy.append(4)
                elif max_index == 'res_gpt4_enhance':
                    s_high_accuracy.append(5)
            return s_high_accuracy

        s_cheapest = [0] * len(self.job_data)
        s_most_expensive = [5] * len(self.job_data)
        # s_high_accuracy = [system_accuracy_map.get(system, 0) for system in self.job_data['System']]
        s_high_accuracy = get_high_accuracy_()

        return s_cheapest, s_most_expensive, s_high_accuracy

    def num_tokens_from_messages(self, messages, model):
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

    def get_level_tokens_num_(self):
        language_models = ["gpt-3.5-turbo", "gpt-4"]
        levels_token = []
        prompt_templates = [
            "You will be provided with a log message delimited by backticks. Please extract the log template from this log message: ",
            "You will be provided with a log message delimited by backticks. You must abstract variables with ‘{placeholders}’ to extract the corresponding template. Print the input log’s template delimited by backticks. Log message: ",
            "You will be provided with a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with ‘{placeholders}‘ and output a static log template. Print the input log’s template delimited by backticks. Log message:",
        ]
        for model in language_models:
            for pt in prompt_templates:
                levels_token.append(self.num_tokens_from_messages(pt, model))

        return levels_token

    def fitness_function_(self, levels_token, s_allocation):
        s_cost = np.zeros(len(s_allocation))
        s_accuracy = np.zeros(len(s_allocation))

        gpt4_input_price = self.model_data.loc[self.model_data['model_name'] == 'gpt-4', 'input_price'].values[0]
        gpt_35_input_price = \
            self.model_data.loc[self.model_data['model_name'] == 'gpt-3.5-turbo', 'input_price'].values[0]

        for i, allocation in enumerate(s_allocation):
            token_num = self.job_data.TokenNum.iloc[i]
            # system = self.job_data['System'][i]
            # system_accuracy = self.df_accuracy.loc[self.df_accuracy['System'] == system].values[0, allocation + 1]

            if allocation <= 2:
                s_cost[i] = (token_num + levels_token[allocation]) * gpt_35_input_price
            else:
                s_cost[i] = (token_num + levels_token[allocation]) * gpt4_input_price

            s_accuracy[i] = self.df_accuracy.iloc[i, allocation]

        s_total_cost = np.sum(s_cost) / 1000
        s_accuracy_mean = np.mean(s_accuracy)
        s_ratio = len(s_accuracy[s_accuracy > 0.8]) / len(s_accuracy)
        # print(s_total_cost, s_accuracy_mean)
        # return [s_total_cost, s_accuracy_mean]
        return [s_total_cost, s_accuracy_mean]

    def run(self):
        start_time = time.time()
        # termination = get_termination("time", self.termination)
        termination = get_termination("n_gen", self.termination)
        s_cheapest, s_most_expensive, s_high_accuracy = self.get_exreme_solution_()
        levels_token = self.get_level_tokens_num_()
        s1_obj1, s1_obj2 = self.fitness_function_(levels_token, s_cheapest)
        s2_obj1, s2_obj2 = self.fitness_function_(levels_token, s_high_accuracy)
        ref_points = np.array([[s1_obj1, -s1_obj2], [s2_obj1, -s2_obj2]])


        n_threads = os.cpu_count()
        # pool = ThreadPool(n_threads)
        # runner = StarmapParallelization(pool.starmap)
        # MyPorblem = AllocationProblemWithReferPoint(job_data=self.job_data, model_data=self.model_data,
        #                                             df_accuracy=self.df_accuracy, levels_token=levels_token,
        #                                             n_var=len(self.job_data), m_max=5, elementwise_runner=runner)

        MyPorblem = AllocationProblemWithReferPoint(job_data=self.job_data, model_data=self.model_data,
                                                    df_accuracy=self.df_accuracy, levels_token=levels_token,
                                                    n_var=len(self.job_data), m_max=5)
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
        r_nsga2_res['true_accuracy'] = get_true_accuracy_(self.job_data, rnsga2_solutions_int)
        print("R-NSGA-2 finished and the searchiing time is: ", elapsed_time)

        return r_nsga2_res, rnsga2_solutions_int #, elapsed_time