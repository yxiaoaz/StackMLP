from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from Baseline_Models_pipeline import load_preprocessed_data,run
import numpy as np

if __name__ == '__main__':
    '''
    print("------------------Multinomial Naive Bayes-------------------")
    df = load_preprocessed_data()
    param_grid= {
        'alpha':[1, 1e-1, 1e-2]
    }
    run(df=df,
        vectorizer=TfidfVectorizer(min_df=5, encoding='utf-8', lowercase=True, max_features=10000, ngram_range=(1, 2)),
        test_size=0.2, over_sampling=True,
        model=MultinomialNB(),
        param_grid=param_grid,
        scoring_metric='roc_auc')

    '''


    print("-------------------------------------------------------------\n------------------Bernoulli Bayes-------------------")
    df = load_preprocessed_data()
    param_grid = {
        'alpha': [1, 1e-1, 1e-2]
    }
    run(df=df,
        vectorizer=TfidfVectorizer(min_df=5, encoding='utf-8', lowercase=True, max_features=10000, ngram_range=(1, 2)),
        test_size=0.2, over_sampling=True,
        model=BernoulliNB(),
        param_grid=param_grid,
        scoring_metric='roc_auc')
    print("-------------------------------------------------------------")

