import json
import optuna
from igs.igs import iterative_greedy_search_
from prediction.prediction_model import *
from prediction.helpers import split_train_test_
from igs.get_true_pf import *
from igs.get_expected_pf import *
from utilities.utils_model import *
from pymoo.algorithms.moo.nsga2 import binary_tournament
from baselines.pymoo_lib import *
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV


def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )


def save_res_(alg, solution):
    fres = create_dir('parameter_setting' + '/' + 'res_')
    with open(fres + alg + '_para' + '.json', 'a', newline='\n') as f:
        json.dump(solution, f)


def parameter_optimization_nsga(trial, df_pre_accuracy, df_true_accuracy, df_cost, model_list, pareto, alg):
    n_var = len(df_true_accuracy)
    m_max = len(df_true_accuracy.iloc[0])

    crossover_prob = trial.suggest_float('crossover_prob', 0.0, 1.0)
    crossover_eta = trial.suggest_int('crossover_eta', 3, 30)
    mutation_prob = trial.suggest_float('mutation_prob', 0.0, 1.0)
    mutation_eta = trial.suggest_int('mutation_eta', 3, 30)

    sampling = trial.suggest_categorical('sampling', ['FloatRandomSampling', 'LHS'])
    if sampling == 'LHS':
        sampling = LHS()
    else:
        sampling = FloatRandomSampling()

    mutation = PolynomialMutation(prob=mutation_prob, eta=mutation_eta)
    crossover = SBX(prob=crossover_prob, eta=crossover_eta)

    if alg == 'NSGA2':
        selection = trial.suggest_categorical('selection', ['RandomSelection', 'TournamentSelection'])
        if selection == 'RandomSelection':
            selection = RandomSelection()
        else:
            selection = TournamentSelection(func_comp=binary_tournament)
        algorithm = NSGA2(pop_size=100,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          selection=selection,
                          eliminate_duplicates=True,
                          )
    elif alg == 'SMS_EMOA':
        algorithm = SMSEMOA(pop_size=100,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            eliminate_duplicates=True,
                            )

    # n_proccess = os.cpu_count()
    # pool = multiprocessing.Pool(n_proccess)
    n_threads = os.cpu_count()
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    MyPorblem = AllocationProblem(df_pre_accuracy, df_true_accuracy, df_cost, model_list, n_var, m_max, elementwise_runner=runner)
    termination = get_termination("n_gen", 100)
    # termination = get_termination("time", "00:00:30")
    res = minimize(MyPorblem,
                   algorithm,
                   termination=termination,
                   seed=1,
                   verbose=False)
    objs = res.F
    objs[:, 1] = objs[:, 1] * -1

    igd = IGD(np.array(pareto))
    igd_res = igd(np.array(objs))

    return igd_res


def parameter_optimization_rnsga2(trial, df_pre_accuracy, df_true_accuracy, df_cost, model_list, pareto, ref_points):
    n_var = len(df_true_accuracy)
    m_max = len(df_true_accuracy.iloc[0])

    epsilon = trial.suggest_float('epsilon', 0.0, 1.0)

    algorithm = RNSGA2(
        ref_points=ref_points,
        pop_size=100,
        epsilon=epsilon,
        normalization='front',
        extreme_points_as_reference_points=True,
        weights=np.array([0.5, 0.5]))

    # n_proccess = os.cpu_count()
    # pool = multiprocessing.Pool(n_proccess)
    n_threads = os.cpu_count()
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    MyPorblem = AllocationProblem(df_pre_accuracy,df_true_accuracy,df_cost, model_list,
                                                    n_var, m_max, elementwise_runner=runner)
    termination = get_termination("n_gen", 100)
    # termination = get_termination("time", "00:00:30")
    res = minimize(MyPorblem,
                   algorithm,
                   termination=termination,
                   seed=1,
                   verbose=False)
    objs = res.F
    objs[:, 1] = objs[:, 1] * -1

    igd = IGD(np.array(pareto))
    igd_res = igd(np.array(objs))

    return igd_res


def classification_tune(trial, train_x, train_y, val_x, val_y, test_x, test_y):


    max_depth = trial.suggest_int('max_depth', 1, 1000)
    n_estimators = trial.suggest_int('n_estimators', 1, 10000)

    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=max_depth, n_estimators=n_estimators))
    clf.fit(train_x, train_y)

    val_pred = clf.predict(val_x[0:])

    # y_pred = clf.predict(test_x[0:])
    return accuracy_score(val_y, val_pred)
    # return accuracy_score(test_y, y_pred)

if __name__ == '__main__':
    with open(f"datasets/log_parsing_word2vec.pkl", mode="rb") as f:
        X, Y, cost = pickle.load(f)

    data_dict = {}
    for i in range(len(X)):
        infor_dict = {}
        infor_dict['features'] = X[i]['features']
        infor_dict['label'] = Y[i]
        infor_dict['cost'] = cost[i]
        data_dict[i] = infor_dict

    train_data, test_data = train_test_split(data_dict, test_size=0.98, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=42)

    train_x, train_y, val_x, val_y, test_x, test_y = split_train_test_(train_data, val_data, test_data)

    study = optuna.create_study(study_name='classification_tuning', load_if_exists=False,
                                    directions=['maximize'],
                                    sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda trial: classification_tune(trial, train_x, train_y, val_x, val_y, test_x, test_y),
        n_trials=100, callbacks=[logging_callback], n_jobs=-1)

    # test_data_size = 0.98
    # data_name = f"datasets/text_classification_word2vec.pkl"
    # # data_name = f"datasets/text_classification/overruling_query_word2vec.pkl"
    # save_dir = f"output/tuning/overruling_{test_data_size}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # model_list = ['gptneox_20B', 'gptj_6B', 'fairseq_gpt_13B', 'text-davinci-002', 'text-curie-001', 'gpt-3.5-turbo',
    #               'gpt-4', 'j1-jumbo', 'j1-grande', 'j1-large', 'xlarge', 'medium']
    # metric = 'igd'  # igd, gd, gdplus, igdplus, time, hv(bigger is better)
    # df_pre_accuracy, df_true_accuracy, df_cost, df_llm_scores, df_llm_ability = data_preprocess(data_name, model_list,
    #                                                                                             test_size=test_data_size)
    #
    #
    #
    #
    # # s_cheapest, s_high_accuracy = iterative_greedy_search_(df_pre_accuracy, df_true_accuracy, df_cost, model_list).get_exreme_solution_()
    # s1_obj = iterative_greedy_search_(df_pre_accuracy, df_true_accuracy, df_cost, model_list).ls_fitness_function_(s_cheapest)
    # s2_obj = iterative_greedy_search_(df_pre_accuracy, df_true_accuracy, df_cost, model_list).ls_fitness_function_(s_high_accuracy)
    # ref_points = np.array([s1_obj, s2_obj])
    # if os.path.exists(f"{save_dir}/true_pt.csv"):
    #     true_pt = pd.read_csv(f"{save_dir}/true_pt.csv")
    # else:
    #     true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
    #     true_pt.to_csv(f"{save_dir}/true_pt.csv")
    #     true_pt_solution = pd.DataFrame(true_pt_solution)
    #     true_pt_solution.to_csv(f"{save_dir}/true_pt_solution.csv")
    #
    #
    # for alg in ["NSGA2", "SMS_EMOA", "RNSGA2"]:
    #     study = optuna.create_study(study_name='algs_para_setting', load_if_exists=False,
    #                                 directions=['minimize'],
    #                                 sampler=optuna.samplers.TPESampler())
    #
    #     if alg == 'RNSGA2':
    #         print('--------------------------Start RNSGA2 optimization--------------------------')
    #         study.optimize(
    #             lambda trial: parameter_optimization_rnsga2(trial, df_pre_accuracy, df_true_accuracy, df_cost,
    #                                                         model_list, true_pt, ref_points),
    #             n_trials=100, callbacks=[logging_callback], n_jobs=-1)
    #
    #         save_res_(alg=alg, solution=study.best_params)
    #
    #         print('best params: ', study.best_params, 'best value: ', study.best_value)
    #         print('--------------------------Finish RNSGA2 optimization--------------------------')
    #     else:
    #         print('--------------------------Start ',alg,' optimization--------------------------')
    #         study.optimize(
    #             lambda trial: parameter_optimization_nsga(trial, df_pre_accuracy, df_true_accuracy, df_cost, model_list, true_pt, alg),
    #             n_trials=100, callbacks=[logging_callback], n_jobs=-1)
    #
    #         save_res_(alg=alg, solution=study.best_params)
    #         print('best params: ', study.best_params, 'best value: ', study.best_value)
    #         print('--------------------------Finish ',alg,' optimization--------------------------')

