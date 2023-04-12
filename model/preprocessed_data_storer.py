from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, PredefinedSplit, GridSearchCV
from imblearn.combine import SMOTEENN
from stack_ensemble_model import *
from dataset_loader import *

def store_preprocessed_data():
    df=pd.read_csv('Stemmed_All.csv', encoding='utf-8')
    to_remove = np.random.choice(df[df['label']==0].index,size=10000,replace=False)
    df=df.drop(to_remove)
    vectorizer=TfidfVectorizer(min_df=5, encoding='utf-8', lowercase=True, max_features=10000, ngram_range=(1, 2))
    vectorized_data = vectorizer.fit_transform(df["textual data"]).toarray()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(vectorized_data, df['label'],df.index, stratify=df['label'],test_size=0.2, random_state=0)
    smt = SMOTEENN(random_state=42)
    X_train, y_train = smt.fit_resample(X_train, y_train)
    with open('resampled_xtrain.npy', 'wb') as f:
        np.save(f, X_train)
    with open('resampled_ytrain.npy', 'wb') as f:
        np.save(f, y_train)
    with open('xtest.npy', 'wb') as f:
        np.save(f, X_test)
    with open('ytest.npy', 'wb') as f:
        np.save(f, y_test)

