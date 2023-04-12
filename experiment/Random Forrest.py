from sklearn.feature_extraction.text import TfidfVectorizer
from Baseline_Models_pipeline import load_preprocessed_data,run
from sklearn.ensemble import RandomForestClassifier
import numpy as np
if __name__ == '__main__':
    df = load_preprocessed_data()
    param_grid= {'bootstrap': [True, False],
             'max_features': ['auto', 'sqrt'],
             'n_estimators': [200, 600, 1000,  1400, 1800]
    }
    run(df=df,
        vectorizer=TfidfVectorizer(min_df=5, encoding='utf-8', lowercase=True, max_features=10000, ngram_range=(1, 2)),
        test_size=0.2, over_sampling=True,
        model=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring_metric='roc_auc')

