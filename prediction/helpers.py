from transformers import BertModel, BertTokenizer
from gensim.models.fasttext import load_facebook_vectors
import numpy as np
from sklearn.model_selection import train_test_split

# model = load_facebook_vectors(
#     "./word_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

# def extract_features_word2vec(text):
#     words = text.strip().split()
#     word_vecs = []
#     for word in words:
#         try:
#             word_vecs.append(model.get_vector(word.strip()))
#         except Exception as ex:
#             pass
#     return np.mean(word_vecs, axis=0)


def split_train_test_(train_data, val_data, test_data):
    train_x = [x['features'] for x in train_data]
    train_y = [list(y['label'].values()) for y in train_data]
    test_x = [x['features'] for x in test_data]
    test_y = [list(y['label'].values()) for y in test_data]
    val_x = [x['features'] for x in val_data]
    val_y = [list(y['label'].values()) for y in val_data]

    return train_x, train_y, val_x, val_y, test_x, test_y

def split_train_test_random(X, Y, test_size):
    # train = data[data['System'] != dataset]
    # test = data[data['System'] == dataset]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    x_train = [x['features'] for x in x_train]
    y_train = [list(y.values()) for y in y_train]
    x_test = [x['features'] for x in x_test]
    y_test = [list(y.values()) for y in y_test]

    return x_train, y_train, x_test, y_test


def split_train_test_dataset(X, Y, dataset):
    x_train, y_train, x_test, y_test = [], [], [], []
    for x, y in list(zip(X, Y)):
        if x['dataset'] == dataset:
            x_test.append(x['features'])
            y_test.append(list(y.values()))
        else:
            x_train.append(x['features'])
            y_train.append(list(y.values()))
    
    return x_train, y_train, x_test, y_test