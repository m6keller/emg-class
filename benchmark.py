import sys
import os

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
CLASSIFIERS = { # Taken from ...
    "SVM":
        {"predictor": "SVM",
            "args": {"C": 100.0, "kernel": "rbf", "degree": 3, "gamma": "auto", "random_state": RANDOM_STATE}},
    "LOR":
        {"predictor": "LOR",
            "args": {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "random_state": RANDOM_STATE}},
    "RF":
        {"predictor": "RF",
            "args": {"n_estimators": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE}},
}
GESTURES = {
    0: "Idle",
    1: "Fist",
    2: "Flexion",
    3: "Extension",
    6: "Pinch index",
    7: "Pinch middle",
    8: "Pinch ring",
    9: "Pinch small"
}

def get_data():
    if len(sys.argv) < 2:
        print("Using hardcoded paths")
        base_data_path = os.path.abspath("./features/subj_4/train_test_splits/04_2018-03-28")
        use_synth = False
        
    else:
        raise Exception("Not implemented")
    
    train_df = pd.read_csv(os.path.join(base_data_path, "train.csv"))
    if use_synth:
        synth_df = pd.read_csv(os.path.join(base_data_path, "synth", "synth_train.csv"))
        train_df = pd.concat([train_df, synth_df])
        
    test_df = pd.read_csv(os.path.join(base_data_path, "test.csv"))
    
    return train_df, test_df
    

def main():
    train_df, test_df = get_data()
    labels = list(filter(lambda col_name: "input" in col_name, train_df.columns))
    
    X_train = train_df[labels]
    y_train = train_df[["TRAJ_GT"]]
    
    X_test = test_df[labels]
    y_test = test_df[["TRAJ_GT"]]
    
    scores = {}
    
    for clf_name, clf_args in CLASSIFIERS.items():
        if clf_name == "SVM":
            clf = SVC(**clf_args["args"])
        elif clf_name == "LOR":
            clf = LogisticRegression(**clf_args["args"])
        elif clf_name == "RF":
            clf = RandomForestClassifier(**clf_args["args"])
            
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"Score for {clf_name}: {score}")
        scores[clf_name] = score
        
    print("Finished running classifiers")
    
    
    
if __name__ == "__main__":
    main()