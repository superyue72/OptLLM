from gensim.models.fasttext import load_facebook_vectors
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle

'''

Extract the features from the FastText word vectors and save them in a pickle file


Download the word vector from https://fasttext.cc/docs/en/english-vectors.html
Pre-trained word vectors:
- wiki-news-300d-1M.vec.zip: 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
- wiki-news-300d-1M-subword.vec.zip: 1 million word vectors trained with subword infomation on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
- crawl-300d-2M.vec.zip: 2 million word vectors trained on Common Crawl (600B tokens).
- crawl-300d-2M-subword.zip: 2 million word vectors trained with subword information on Common Crawl (600B tokens) (** current one **)
'''



def extract_features_word2vec(text):
    words = text.strip().split()
    word_vecs = []
    for word in words:
        try:
            word_vecs.append(model.get_vector(word.strip()))
        except Exception as ex:
            pass
    return np.mean(word_vecs, axis=0)


if __name__ == '__main__':
    # Load the FastText model from the file
    model = load_facebook_vectors(
        "./word_models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

    data_dir = "../datasets/text_classification/"   # data_dir = "../datasets/log_parsing/"
    model_names = ["gptneox_20B", "gptj_6B", "fairseq_gpt_13B", "text-davinci-002", "text-curie-001",
                   "gpt-3.5-turbo", "gpt-4", "j1-jumbo", "j1-grande", "j1-large", "xlarge", "medium"]
    # model_names = ["j2_mid", "j2_ultra", "Mixtral_8x7B", "llama2_7b", "llama2_13b", "llama2_70b", "Yi_34B", "Yi_6B"]

    answer_column = "ref_answer"
    query_name = "content"

    # single dataset preprocess
    datasets = ['agnews', 'coqa', 'headlines', 'overruling', 'sciq']
    for dataset in datasets:
        job_data = pd.read_csv(f"{data_dir}/{dataset}.csv")
        X = []
        Y = []
        cost = []
        for _, row in job_data.iterrows():
            X.append({'dataset': dataset, 'query': row[query_name]})
            Y.append({k: row[f"{k}_answer"] == row[answer_column] for k in model_names})
            cost.append({k: row[f"{k}_cost"] for k in model_names})
        X_features_word2vec = [{'dataset': x['dataset'], 'features': extract_features_word2vec(x['query'])} for x in X]
        Y = [{k: 1 if v else 0 for k, v in y.items()} for y in Y]
        print(dataset, len(job_data))
        with open(f"{dataset}_content_word2vec.pkl", mode="wb") as f:
            pickle.dump((X_features_word2vec, Y, cost), f)