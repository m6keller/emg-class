"""
Generate splits to automatically save to CSV within subject folder. 
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
        splits_folder = os.path.abspath("./features/subj_4/train_test_splits")
        calculated_features_folder = os.path.join(result_folder, 'calculated_features')
    else: 
        base_folder_for_subject = os.path.abspath(sys.argv[1])
        calculated_features_folder = os.path.join(base_folder_for_subject, "calculated_features")
        splits_folder = os.path.join(base_folder_for_subject, "splits")
        data_folder = os.path.join(base_folder_for_subject, "data")
        
    for folder in [calculated_features_folder, splits_folder, data_folder]:
        if not os.path.isdir(folder):
            os.mkdir(folder) 
    
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
    train_test_splits = { id_date: splits_list[0] for id_date, splits_list in splits_all.items() }
    
    for id_, id_splits in train_test_splits.items():
        print('\tTrial ID: {:s}'.format(id_), flush=True)

        cur_split_folder = os.path.join(splits_folder, id_.replace('/', '_'))
        if not os.path.exists(cur_split_folder):
            os.mkdir(cur_split_folder)
            
        # for split in k-fold validation of each day of each subject
        # for i_s, s in enumerate(id_splits):
        data = biolab_utilities.prepare_data(dfs, id_splits, FEATURE_SET, list(GESTURES.keys()))
        train_df = pd.DataFrame(data["train"])
        test_df = pd.DataFrame(data["test"])
        print(f"Saving splits to csv...", flush=True)
        save_train_path = os.path.join(cur_split_folder, 'train.csv')
        save_test_path = os.path.join(cur_split_folder, 'test.csv')
        train_df.to_csv(save_train_path)
        test_df.to_csv(save_test_path)
        print("Saved to: ", cur_split_folder, flush=True)
            
    print("Saved all splits to: ", splits_folder, flush=True)
    

if __name__ == "__main__":
    main()