"""
Generates new features from path to calculated_features for set of filtered EMG features 
"""

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
    
    
    pass

train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

def main():
    # generate data
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
