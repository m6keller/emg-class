"""
Generates new features from path to calculated_features for set of filtered EMG features 
"""


import os
import sys
import glob
import pickle
import re
from typing import List, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix

import putemg_features
from putemg_features import biolab_utilities

from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np


def get_data():
    """
    Returns:
        train: pd.DataFrame
        target: pd.DataFrame
        test: pd.DataFrame
    """
    
    if len(sys.argv) < 3:
        print('Using Hard Coded paths')
        putemg_folder = os.path.abspath("./features/subj_4/data")
        result_folder = os.path.abspath("./features/subj_4")
        # putemg_folder = os.path.abspath("./benchmark_features")

    else: 
        putemg_folder = os.path.abspath(sys.argv[1])
        result_folder = os.path.abspath(sys.argv[2])

        if not os.path.isdir(putemg_folder):
            print('{:s} is not a valid folder'.format(putemg_folder))
            exit(1)

        if not os.path.isdir(result_folder):
            print('{:s} is not a valid folder'.format(result_folder))
            exit(1)
    
    
    calculated_features_folder = os.path.join(result_folder, 'calculated_features')

    # list all hdf5 files in given input folder
    all_files = [f for f in sorted(glob.glob(os.path.join(putemg_folder, "*.hdf5")))]
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


    splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=3)


    train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
    target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
    test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

def main():
    # generate data
    train, target, test = get_data()
    new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
    new_train2, new_target2 = GANGenerator().generate_data_pipe(train, target, test, )

    # example with all params defined
    new_train3, new_target3 = GANGenerator(gen_x_times=1.1, cat_cols=None,
            bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=True,
            adversarial_model_params={
                "metrics": "AUC", "max_depth": 2, "max_bin": 100, 
                "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
            }, pregeneration_frac=2, only_generated_data=False,
            gan_params = {"batch_size": 500, "patience": 25, "epochs" : 500,}).generate_data_pipe(train, target,
                                            test, deep_copy=True, only_adversarial=False, use_adversarial=True)

if __name__ == "__main__":
    main()