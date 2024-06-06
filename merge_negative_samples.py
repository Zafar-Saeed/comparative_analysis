import os
import constants
import Util
import json

root_path = "/source_codes/kge-rl/data"
#root_path = "./data"
def load_and_merge_negatives():
    if os.path.exists(os.path.join(root_path,"experiment_specs","test_exp_config.json")):
        main_config = json.load(open(os.path.join(root_path,"experiment_specs","test_exp_config.json")))
        if os.path.exists(os.path.join(root_path,main_config["exp_name"],"config.json")):
            constants.experiment_config = json.load(open(os.path.join(root_path,main_config["exp_name"],"config.json")))
        else:
            raise Exception("Experiment configuration file does not exist.")    
    else:
        raise Exception("Main configuration file does not exist.")

    if os.path.exists(os.path.join(constants.data_path,main_config["exp_name"],
        "".join([constants.negative_sample_file_prefix, "_FULL.pkl"]))):
            print("Pickel file already existed..")
            # do nothing here, because the negative sampler will later load the file
    else:
        full_ns_dictionary = Util.merge_negative_sample_files(
        directory_path=os.path.join(root_path,main_config["exp_name"]),
        file_prefix=constants.negative_sample_file_prefix)

        Util.dump_dict_to_pickle_file(
            full_ns_dictionary,
            os.path.join(constants.data_path,main_config["exp_name"],
                        "".join([constants.negative_sample_file_prefix, "_FULL.pkl"]))
        )

load_and_merge_negatives()
