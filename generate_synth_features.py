"""
Generates new features from path to calculated_features for set of filtered EMG features 
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Need to run this script with python3 - cannot get tabgan working with Python 2
from tabgan.sampler import GANGenerator

def get_data():
    if len(sys.argv) < 2:
        print("Not enough arguments provided")
        # exit(1)
        print("Using hardcoded paths")
        base_path = os.path.abspath("./features/subj_4/train_test_splits/04_2018-03-28")
    else:
        base_path = os.path.abspath(sys.argv[1])
    
    train_path = os.path.join(base_path, "train.csv")
    
    if not os.path.isfile(train_path):
        print("Train file does not exist")
        exit(1)
        
    return pd.read_csv(train_path), base_path
    
        
        

def main():
    # generate data
    data, base_path = get_data()
    labels = list(filter(lambda col_name: "input" in col_name, data.columns))
    
    all_features = data[labels]
    all_targets = data[["TRAJ_GT"]]

    train_x, test_x, train_y, _ = train_test_split(all_features, all_targets, test_size=0.2, random_state=42)
    
    new_train, new_target = GANGenerator().generate_data_pipe(train_x, train_y, test_x)
    
    synth_data_df = pd.DataFrame(new_train, columns=labels)
    synth_data_df["TRAJ_GT"] = new_target

    synth_output_dir = os.path.join(base_path, "synth")
    if not os.path.isdir(synth_output_dir):
        os.mkdir(synth_output_dir)
        
    output_file = os.path.join(synth_output_dir, "synth_train.csv")

    synth_data_df.to_csv(output_file, index=False)
    
    
if __name__ == "__main__":
    main()