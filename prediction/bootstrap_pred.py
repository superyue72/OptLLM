import numpy as np
import pandas as pd
import torch
import os
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from baselines.pygmo_lib import *
def robust_prediction(train_x, train_y, val_x, val_y, test_x):
    n_bootstraps = 100  # Number of bootstrap samples
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    n_outputs = train_y.shape[1]
    bootstrap_preds = np.zeros((len(test_x), n_outputs, n_bootstraps))
    best_estimator = None
    best_val_score = -np.inf

    for i in range(n_bootstraps):
        # Resample the training data with replacement
        train_x_boot, train_y_boot = resample(train_x, train_y)

        base_estimator = MultiOutputClassifier(
            estimator=RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
        )
        base_estimator.fit(train_x_boot, train_y_boot)

        # Evaluate the model on the validation set
        val_preds = base_estimator.predict_proba(val_x)

        val_preds_array = []
        for pred in val_preds:
            if pred.shape[1] == 2:
                val_preds_array.append(pred[:, 1])
            else:
                val_preds_array.append(pred.ravel())
        val_preds_array = np.stack(val_preds_array, axis=1)

        # val_preds_array = np.stack(val_preds, axis=1)[:, :, 1]
        val_score = np.mean(val_preds_array == val_y)

        # Update the best estimator if the validation score improves
        if val_score > best_val_score:
            best_estimator = base_estimator
            best_val_score = val_score

        # Predict on the test set and store the predictions
        preds_list = base_estimator.predict_proba(test_x)
        preds_array = np.stack(preds_list, axis=1)[:, :, 1]
        bootstrap_preds[:, :, i] = preds_array

    robust_values = np.mean(bootstrap_preds, axis=2)  # Mean predictions
    uncertainty = np.std(bootstrap_preds, axis=2)  # Standard deviation

    return robust_values, uncertainty
def split_data(data):
    data_x = [x['features'] for x in data]
    data_y = [list(y['label'].values()) for y in data]
    return data_x, data_y

def get_llm_score_(val_y, val_pred, model_list):
    report = classification_report(val_y, val_pred, digits=3, target_names=model_list)
    lines = report.split('\n')
    scores_data = []
    for line in lines[2:-5]:  # Skip the first 2 and last 5 lines which contain headers and averages
        row_data = line.split()
        if len(row_data) < 2 or row_data[0] == 'micro' or row_data[0] == 'macro' or row_data[0] == 'weighted':
            continue  # Skip lines that are not relevant to individual classes
        class_name = row_data[0]
        precision = float(row_data[1])
        recall = float(row_data[2])
        f1_score = float(row_data[3])
        support = int(row_data[4])
        scores_data.append([class_name, precision, recall, f1_score, support])
    scores_df = pd.DataFrame(scores_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    scores_df.set_index('Class', inplace=True)
    return scores_df

def get_llm_ability_(train_data, val_data, model_list):
    train_label = pd.DataFrame([x['label'] for x in train_data])
    val_label = pd.DataFrame([x['label'] for x in val_data])
    data = pd.concat([train_label, val_label])
    score = pd.DataFrame(data.sum(axis=0) / len(data), columns=['score'])
    score["model"] = score.index
    score = score.reset_index(drop=True)

    df_sorted = score.sort_values(by='score', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted['order'] = df_sorted.index + 1
    df_sorted_custom_order = df_sorted.set_index('model').loc[model_list].reset_index()
    df_sorted_custom_order.set_index('model', inplace=True, drop=True)
    return df_sorted_custom_order
def get_predict_accuracy_(data_dir, model_list, itr, alpha, test_size):
    acc_table = {}
    with open(data_dir, mode="rb") as f:
        X, Y, cost = pickle.load(f)

    data_dict = {}
    for i in range(len(X)):
        infor_dict = {}
        infor_dict['features'] = X[i]['features']
        infor_dict['label'] = Y[i]
        infor_dict['cost'] = cost[i]
        data_dict[i] = infor_dict

    train_data, test_data = train_test_split(data_dict, test_size=test_size, random_state=int(itr*2+5))
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=int(itr*3+1))

    train_x, train_y = split_data(train_data)
    val_x, val_y = split_data(val_data)
    test_x, test_y = split_data(test_data)

    clf = MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1))
    clf.fit(train_x, train_y)

    y_score = clf.predict_proba(test_x)
    y_pred = clf.predict(test_x[0:])

    original_acc = accuracy_score(test_y, y_pred)
    acc_table['original'] = original_acc

    val_score = clf.predict_proba(val_x[0:])
    y_val_pred = [[y_score[j][i][1] for j in range(len(model_list))] for i in range(len(val_score[0]))]
    val_pred = clf.predict(val_x[0:])
    df_llm_scores = get_llm_score_(val_y, val_pred, model_list)

    test_robust_values, test_robust_uncertainties = robust_prediction(train_x, train_y, val_x, y_val_pred, test_x)

    for a in [-1, -0.5, 0, 0.5, 1]:
        robust_values = test_robust_values + a * test_robust_uncertainties
        robust_y = np.where(robust_values > 0.5, 1, 0)
        robust_acc = accuracy_score(test_y, robust_y)
        acc_table[a] = robust_acc

    test_robust_values = test_robust_values + alpha * test_robust_uncertainties
    robust_y = np.where(test_robust_values > 0.5, 1, 0)


    complexity = [[y_score[j][i][1] for j in range(len(model_list))] for i in range(len(y_score[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)
    # df_pre_accuracy = pd.DataFrame(y_pred, columns=model_list)
    df_true_accuracy = pd.DataFrame(test_y, columns=model_list)


    print(classification_report(test_y, robust_y, digits=3, target_names=model_list))
    print(accuracy_score(test_y, robust_y))

    df_robust_accuracy = pd.DataFrame(test_robust_values, columns=model_list)

    text_cost = [x['cost'] for x in test_data]
    df_cost = pd.DataFrame(text_cost)

    llm_ability = get_llm_ability_(train_data, val_data, model_list)

    return df_pre_accuracy, df_robust_accuracy, df_true_accuracy, df_llm_scores, llm_ability, df_cost, acc_table


