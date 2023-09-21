"""
Generate splits to automatically save to CSV within subject folder. Generates K-fold validation splits for each subject. 
"""

import os
import sys
import glob
from typing import Dict

import pandas as pd
from putemg_features import biolab_utilities

import pandas as pd

N_SPLITS = 3
FEATURE_SET = ["RMS", "MAV", "WL", "ZC", "SSC",]
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

def parse_arguments():
    if len(sys.argv) < 2:
        # print("Not enough arguments")
        # exit(1)
        print('Using Hard Coded paths')
        data_folder = os.path.abspath("./features/subj_4/data")
        result_folder = os.path.abspath("./features/subj_4/")
        splits_folder = os.path.abspath("./features/subj_4/splits")
        calculated_features_folder = os.path.join(result_folder, 'calculated_features')
    else: 
        base_folder_for_subject = os.path.abspath(sys.argv[1])
        calculated_features_folder = os.path.join(base_folder_for_subject, "calculated_features")
        splits_folder = os.path.join(base_folder_for_subject, "splits")
        data_folder = os.path.join(base_folder_for_subject, "data")
        
    if not os.path.isdir(data_folder):
        print('{:s} is not a valid folder'.format(data_folder))
        exit(1)
    
    return calculated_features_folder, splits_folder, data_folder

        
    

def main():
    calculated_features_folder, splits_folder, data_folder = parse_arguments()
    
    # list all hdf5 files in given input folder
    all_files = [f for f in sorted(glob.glob(os.path.join(data_folder, "*.hdf5")))]
    all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]

    # data can be additionally filtered based on subject id
    records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records)

    # load feature data to memory
    dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
    for r in records_filtered_by_subject:
        print("Reading features for input file: ", r)
        filename = os.path.splitext(r.path)[0]
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder,
                                                       filename + '_filtered_features.hdf5')))


    splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=N_SPLITS)
    
    for id_, id_splits in splits_all.items():
        print('\tTrial ID: {:s}'.format(id_), flush=True)

        # for split in k-fold validation of each day of each subject
        for i_s, s in enumerate(id_splits):
            data = biolab_utilities.prepare_data(dfs, s, FEATURE_SET, list(GESTURES.keys()))
            train_df = pd.DataFrame(data["train"])
            test_df = pd.DataFrame(data["test"])
            
            dir_for_cur_split = os.path.join(splits_folder, f"id_{str(i_s)}")
            if os.path.exists(dir_for_cur_split):
                print("Directory already exists. Skipping...")
            else:
                print("Creating directory: ", dir_for_cur_split)
                os.mkdir(dir_for_cur_split)
            print(f"Saving splits to csv for ID f{i_s}...", flush=True)
            save_train_path = os.path.join(dir_for_cur_split, 'train.csv')
            save_test_path = os.path.join(dir_for_cur_split, 'test.csv')
            train_df.to_csv(save_train_path)
            test_df.to_csv(save_test_path)
            print("Saved to: ", dir_for_cur_split, flush=True)
            
    print("Testing after saving")
    

if __name__ == "__main__":
    main()