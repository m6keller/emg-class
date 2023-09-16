import sys
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def calculate_metrics(y_true, y_pred):
    # Calculate True Positives, True Negatives, False Positives, and False Negatives
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    true_negatives = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    false_negatives = np.sum(np.logical_and(y_true == 1, y_pred == 0))

    # Calculate Accuracy
    accuracy = (true_positives + true_negatives) / len(y_true)

    # Calculate Sensitivity (True Positive Rate or Recall)
    sensitivity = true_positives / (true_positives + false_negatives)

    # Calculate Specificity (True Negative Rate)
    specificity = true_negatives / (true_negatives + false_positives)

    return accuracy, sensitivity, specificity

def main():
    if len(sys.argv) < 2:
        print("Using hardcoded values")
        rel_path_to_output_file = "features/subj_4/classification_result_8chn_2band.bin"
    else:
        rel_path_to_output_file = sys.argv[1]

    path_to_output_file = os.path.abspath(rel_path_to_output_file)
    
    output: Dict[str, any] = pickle.load(open(path_to_output_file, "rb"))
    
    results_dfs = {}
    id_to_cm = {}
    
    result_with_highest_acc = None
    highest_acc = 0

    for result in output["results"]:
        # results_dfs[f"{result['clf']}_{result[id]}"] = pd.DataFrame({
        #     "y_true": result["y_true"],
        #     "y_pred": result["y_pred"],
        # })
        
        acc, sens, spec = calculate_metrics(result["y_true"], result["y_pred"])
        
        print(f"ID: {result['id']}")
        print(f"Classifier: {result['clf']}")
        
        print(f"Accuracy: {acc}")
        print(f"Sensitivity: {sens}")
        print(f"Specificity: {spec}")
        
        print()
        if acc > highest_acc:
            result_with_highest_acc = result
            highest_acc = acc
    
    print("---------------------------------")
    print("HIGHEST ACCURACY RESULT:")
    print(f"ID: {result_with_highest_acc['id']}")
    print(f"Classifier: {result_with_highest_acc['clf']}")
    print(f"Accuracy: {highest_acc}")
    acc, sens, spec = calculate_metrics(result_with_highest_acc["y_true"], result_with_highest_acc["y_pred"])
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    
    # plt.plot(results_df["epoch"], results_df["loss"])

    pass

if __name__ == "__main__":
    main()

