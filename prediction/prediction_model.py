import pandas as pd
import pickle
from gensim.models.fasttext import load_facebook_vectors
from prediction.helpers import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

def get_predict_accuracy(model_list, train_data, test_data):
    train_x, train_y, test_x, test_y = split_train_test_(train_data, test_data)
    clf = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1, max_depth=100, n_estimators=1000))
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x[0:])
    y_score = clf.predict_proba(test_x[0:])

    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(test_y, y_pred, digits=3, target_names=model_list))
    print(accuracy_score(test_y, y_pred))
    complexity = [[y_score[j][i][1] for j in range(len(model_list))] for i in range(len(y_score[0]))]
    df_pre_accuracy = pd.DataFrame(complexity, columns=model_list)
    # df_pre_accuracy = pd.DataFrame(y_pred, columns=model_list)
    df_true_accuracy = pd.DataFrame(test_y, columns=model_list)
    return df_pre_accuracy, df_true_accuracy


def data_preprocess(data_dir, dataset, model_list, test_size=0.99):
    print(test_size)
    with open(f"{data_dir}/{dataset}_query_word2vec.pkl", mode="rb") as f:
        X, Y, cost = pickle.load(f)

    data_dict = {}
    for i in range(len(X)):
        infor_dict = {}
        infor_dict['features'] = X[i]['features']
        infor_dict['label'] = Y[i]
        infor_dict['cost'] = cost[i]
        data_dict[i] = infor_dict

    train_data, test_data = train_test_split(data_dict, test_size=test_size, random_state=42)
    df_pre_accuracy, df_true_accuracy = get_predict_accuracy(model_list, train_data, test_data)
    text_cost = [x['cost'] for x in test_data]
    df_cost = pd.DataFrame(text_cost)

    return df_pre_accuracy, df_true_accuracy, df_cost
