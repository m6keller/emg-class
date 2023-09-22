import putemg_features

import sys
import os
import pandas as pd

PATH_TO_ALL_FEATURES = "./features"
PATH_TO_RAW_DATA = "./Data-HDF5"
PATH_TO_FEATURES_XML = "./benchmark_features.xml"

def calc_and_save_features(path_to_raw_data_hdft: str, path_to_features_xml: str, output_path: str):  
    features_df = putemg_features.features_from_xml(xml_file_url=path_to_features_xml, hdf5_file_url=path_to_raw_data_hdft)
    features_df.to_hdf(output_path, 'data', format='table', mode='w', complevel=5)

def main():
    
    subj_id = sys.argv[1]
    
    if not subj_id.isdigit() :
        print("Need to input valid ID")
        exit(1)
            
    subj_output_path = os.path.abspath(os.path.join(PATH_TO_ALL_FEATURES, f"subj_{subj_id}"))
        
    features_path = os.path.join(subj_output_path, "calculated_features")
    
    for path in [subj_output_path, features_path]:
        if not os.path.isdir(path):
            os.mkdir(path)
    
    all_raw_data_paths = os.listdir(PATH_TO_RAW_DATA)
    
    raw_data_paths_for_subj = list(filter(lambda filename: f"emg_gestures-{subj_id.zfill(2)}" in filename, all_raw_data_paths))
    for file_path in raw_data_paths_for_subj:
        output_file_name = file_path.rstrip("hdf5") + "_filtered_features.hdf5"
        output_path = os.path.join(features_path, output_file_name)
        calc_and_save_features(path_to_raw_data_hdft=os.path.join(PATH_TO_RAW_DATA, file_path), path_to_features_xml=PATH_TO_FEATURES_XML, output_path=output_path)
        
    print("Finished")
    
if __name__ == "__main__":
    main()
