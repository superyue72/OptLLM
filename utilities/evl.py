import numpy as np
import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume


def calculate_igd_metric(solution_set, pareto_set):
    # gd = GD(pareto_set)
    igd = IGD(pareto_set)
    return igd(solution_set)


def calculate_hv_metric(solution_set, pareto_set):
    def get_refer_point_(pareto_set):
        middle_point = np.median(pareto_set, axis=0)
        return middle_point

    middle_point = get_refer_point_(pareto_set)
    approx_ideal = solution_set.min(axis=0)
    approx_nadir = solution_set.min(axis=0)
    hv = Hypervolume(ref_point=np.array(middle_point), norm_ref_point=False, zero_to_one=True, ideal=approx_ideal,
                     nadir=approx_nadir)
    return hv(np.array(solution_set))


def calculate_spacing_metric(population):
    num_solutions, num_objectives = population.shape

    if num_solutions <= 1:
        return 0.0

    # Initialize spacing sum and min_distance array
    spacing_sum = 0.0
    min_distances = np.zeros(num_solutions)

    # Calculate the minimum distances for each solution
    for i in range(num_solutions):
        min_distance = np.inf
        for j in range(num_solutions):
            if i != j:
                distance = np.sqrt(np.sum((population[i] - population[j]) ** 2))
                min_distance = min(min_distance, distance)
        min_distances[i] = min_distance

    # Calculate the mean distance
    mean_distance = np.mean(min_distances)

    # Calculate the squared differences and sum them
    squared_diff_sum = np.sum((mean_distance - min_distances) ** 2)

    # Calculate the spacing metric
    spacing_metric = np.sqrt(squared_diff_sum / (num_solutions - 1))

    return spacing_metric


def calculate_delta_metric(population, pareto_front):
    num_solutions, num_objectives = population.shape

    # Calculate Euclidean distances between extreme solutions on the Pareto front
    d_f = np.linalg.norm(pareto_front[0] - population[0])
    d_l = np.linalg.norm(pareto_front[-1] - population[-1])

    # Calculate mean distance between consecutive solutions in the population
    distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
    mean_distance = np.mean(distances)

    # Calculate distances between solutions and their nearest neighbors
    min_distances = []
    for i in range(num_solutions):
        other_solutions = np.delete(population, i, axis=0)
        distances_to_others = np.linalg.norm(other_solutions - population[i], axis=1)
        min_distances.append(np.min(distances_to_others))

    # Calculate the Delta metric
    delta_metric = (d_f + d_l + np.sum(np.abs(min_distances - mean_distance))) / (
                d_f + d_l + (num_solutions - 1) * mean_distance)

    return delta_metric


def get_avg_metric_(alg_res, true_pt):
    all_igd = []
    all_delta = []
    all_time = []
    all_n = []
    for itr in range(1, 11):
        all_time.append(alg_res.iloc[0]['time'] / 60)
        res = pd.DataFrame({'cost': alg_res['cost'], 'true_accuracy': alg_res['true_accuracy']})
        np_res = np.array(res)
        np_true_pt = np.array(true_pt.copy())
        all_igd.append(calculate_igd_metric(np_res, np_true_pt))
        all_delta.append(calculate_delta_metric(np_res, np_true_pt))
        all_n.append(len(np_res))
        rounded_igd_mean = round(np.mean(all_igd), 2)
        rounded_delta_mean = round(np.mean(all_delta), 2)
        rounded_time_mean = round(np.mean(all_time), 2)
        rounded_n = round(np.mean(all_n), 0)

    return rounded_igd_mean, rounded_delta_mean, rounded_time_mean, rounded_n  # , rounded_spacing_mean, rounded_hv_mean

