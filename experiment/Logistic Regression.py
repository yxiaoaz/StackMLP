from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from Baseline_Models_pipeline import load_preprocessed_data,run
import numpy as np

if __name__ == '__main__':
    df = load_preprocessed_data()
    param_grid= {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 50)
    }
    run(df=df,
        vectorizer=TfidfVectorizer(min_df=5, encoding='utf-8', lowercase=True, max_features=10000, ngram_range=(1, 2)),
        test_size=0.2, over_sampling=True,
        model=LogisticRegression(random_state=42),
        param_grid=param_grid,
        scoring_metric='roc_auc')
