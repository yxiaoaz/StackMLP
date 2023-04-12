import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocessing


def load_preprocessed_data():
    if os.path.exists('Stemmed_All.csv'):
        df = pd.read_csv('Stemmed_All.csv', encoding='utf-8')  # ./Project./All.csv
    else:
        df=preprocessing("All.csv").run()
    return df

def run(df,vectorizer,test_size,over_sampling,model,param_grid,scoring_metric):
    print("Starts pipeline")

    vectorized_data = vectorizer.fit_transform(df["textual data"]).toarray()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(vectorized_data, df['label'],
                                                                                     df.index, stratify=df['label'],
                                                                                   test_size=test_size, random_state=0)
    if over_sampling:
        random_state = 42
        smt = SMOTEENN(random_state=random_state)
        X_train, y_train = smt.fit_resample(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    model_clf = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring=scoring_metric,
                          cv=cv,
                          verbose=0,
                          n_jobs=1)

    y_pred_acc = model_clf.fit(X_train, y_train).predict(X_test)
    print("-------Best Model Parameters:--------")
    print(model_clf.best_params_)
    print("-------------------------------------")
    # New Model Evaluation metrics
    print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred_acc)))
    print('Precision Score : ' + str(precision_score(y_test, y_pred_acc)))
    print('Recall Score : ' + str(recall_score(y_test, y_pred_acc)))
    print('F1 Score : ' + str(f1_score(y_test, y_pred_acc)))
    print("AUC: "+str(roc_auc_score(y_test,y_pred_acc)))

    # Logistic Regression (Grid Search) Confusion matrix

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_acc))
    disp.plot()
    plt.show()


