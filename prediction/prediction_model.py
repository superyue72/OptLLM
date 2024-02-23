import pandas as pd
import pickle
import string
from gensim.models.fasttext import load_facebook_vectors
from prediction.helpers import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier


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


def get_predict_accuracy_and_score_(model_list, train_data, val_data, test_data):
    train_x, train_y, val_x, val_y, test_x, test_y = split_train_test_(train_data, val_data, test_data)
    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=100, n_estimators=1000))
    clf.fit(train_x, train_y)

    val_pred = clf.predict(val_x[0:])
    # val_score = clf.predict_proba(val_x[0:])
    df_llm_scores = get_llm_score_(val_y, val_pred, model_list)

    y_pred = clf.predict(test_x[0:])
    y_score = clf.predict_proba(test_x[0:])

    print(classification_report(test_y, y_pred, digits=3, target_names=model_list))
    print(accuracy_score(test_y, y_pred))
    complexity = [[y_score[j][i][1] for j in range(len(model_list))] for i in range(len(y_score[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)
    # df_pre_accuracy = pd.DataFrame(y_pred, columns=model_list)
    df_true_accuracy = pd.DataFrame(test_y, columns=model_list)
    return df_pre_accuracy, df_true_accuracy, df_llm_scores


def data_preprocess(data_dir, model_list, itr, test_size=0.98):
    print(test_size)
    with open(data_dir, mode="rb") as f:
        X, Y, cost = pickle.load(f)
    # with open(f"{data_dir}/{dataset}_query_word2vec.pkl", mode="rb") as f:
    #     X, Y, cost = pickle.load(f)

    data_dict = {}
    for i in range(len(X)):
        infor_dict = {}
        infor_dict['features'] = X[i]['features']
        infor_dict['label'] = Y[i]
        infor_dict['cost'] = cost[i]
        data_dict[i] = infor_dict

    train_data, test_data = train_test_split(data_dict, test_size=test_size, random_state=itr)
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=itr)

    print("There are", len(train_data), "train data and ", len(val_data), "val data and ", len(test_data), "test data")

    df_pre_accuracy, df_true_accuracy, df_llm_scores = get_predict_accuracy_and_score_(model_list, train_data, val_data,
                                                                                       test_data)
    text_cost = [x['cost'] for x in test_data]
    df_cost = pd.DataFrame(text_cost)

    llm_ability = get_llm_ability_(train_data, val_data, model_list)

    return df_pre_accuracy, df_true_accuracy, df_cost, df_llm_scores, llm_ability



def log_preprocess(data_dir, model_list, test_size=0.99):
    def features_extraction(log):
        features = []
        # number of tokens
        features.append(len(log.split()))
        # number of unique tokens
        features.append(len(set(log.split())))
        # number of characters
        features.append(len(log))
        # number of unique characters
        features.append(len(set(log)))
        # number of digits
        features.append(sum(c.isdigit() for c in log))
        # number of letters
        features.append(sum(c.isalpha() for c in log))
        # number of punctuations
        features.append(sum(c in string.punctuation for c in log))
        # average number of characters per token
        features.append(features[2] / features[0])
        # average number of characters per unique token
        features.append(features[2] / features[1])
        # average number of digits per token
        features.append(features[4] / features[0])
        # average number of punctuations per token
        features.append(features[6] / features[0])
        # max length of token
        features.append(max(len(token) for token in log.split()))
        # min length of token
        features.append(min(len(token) for token in log.split()))
        # max punctuation length of token
        features.append(max(sum(c in string.punctuation for c in token) for token in log.split()))
        # min punctuation length of token
        features.append(min(sum(c in string.punctuation for c in token) for token in log.split()))
        # max digit length of token
        features.append(max(sum(c.isdigit() for c in token) for token in log.split()))
        # min digit length of token
        features.append(min(sum(c.isdigit() for c in token) for token in log.split()))
        return features
    def get_cost_(job_data, model_list):
        cost_data = pd.DataFrame()
        for model in model_list:
            cost_column = f'{model}_cost'
            cost_data[model] = job_data[cost_column]
        return cost_data

    def get_accuracy_(job_data, model_list):
        accuracy_data = pd.DataFrame()
        for model in model_list:
            cost_column = f'{model}'
            accuracy_data[model] = job_data[cost_column]
        return accuracy_data

    all_data = pd.read_csv(data_dir, index_col=0)
    for model in model_list:
        all_data[model] = all_data['ref_answer'] == all_data[f'{model}_answer']

    train_data, job_data = train_test_split(all_data, test_size=test_size, random_state=42)

    model_num = len(model_list)
    def process_row(row):
        x = features_extraction(row['content'])  # Assuming 'query' is the input feature
        y = [row[model] for model in model_list]
        return x, y

    train_x, train_y = zip(*train_data.apply(process_row, axis=1))
    test_x, test_y = zip(*job_data.apply(process_row, axis=1))

    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=10, n_estimators=1000))
    clf.fit(train_x, train_y)

    y_pred_accuracy = clf.predict_proba(test_x)
    y_pred = clf.predict(test_x)
    complexity = [[y_pred_accuracy[j][i][1] for j in range(model_num)] for i in range(len(y_pred_accuracy[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)


    print(classification_report(np.array(test_y), y_pred, digits=3, target_names=model_list))
    print('Accuracy Score: ', accuracy_score(np.array(test_y), y_pred))

    df_cost = get_cost_(job_data, model_list)
    df_true_accuracy = get_accuracy_(job_data, model_list)

    return df_pre_accuracy, df_true_accuracy, df_cost