# from pykeen.pipeline import pipeline
# from pykeen.datasets import get_dataset
# # from pykeen.datasets import Freebase15
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd

# # # Load the Freebase15K dataset
# # dataset = Freebase15()

# # # Split the dataset into train, validation, and test sets
# # train_valid, test = train_test_split(dataset.training.to_pd_dataframe(), test_size=0.2, random_state=1234)
# # train, valid = train_test_split(train_valid, test_size=0.2, random_state=1234)
# dataset_path = "F:/University of Bari - Italy/Research/Source Code and Datasets/Datasets/Roberto Datasets/FB15k"
# # Load train, validation, and test datasets from text files
# train_file = dataset_path + "/train.txt"
# valid_file = dataset_path + "/valid.txt"
# test_file = dataset_path + "/test.txt"

# train_df = pd.read_csv(train_file, sep='\t', names=['head_label', 'relation_label', 'tail_label'])
# valid_df = pd.read_csv(valid_file, sep='\t', names=['head_label', 'relation_label', 'tail_label'])
# test_df = pd.read_csv(test_file, sep='\t', names=['head_label', 'relation_label', 'tail_label'])


# # Define the pipeline
# results = pipeline(
#     # training=train,
#     # testing=test,
#     # validation=valid,
#     training=train_df,
#     testing=valid_df,
#     validation=test_df,
    
#     model='TransE',
#     negative_sampler='Bernoulli',
#     training_kwargs=dict(num_epochs=100, batch_size=128),
#     evaluation_kwargs=dict(),
#     model_kwargs=dict(embedding_dim=50),
#     random_seed=1234,
#     device='cpu',  # Change to 'cuda' if you have GPU available
# )

# # Print the evaluation results
# print(results)





###### WITH TRIPLE FACTORY #####

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
import json
import os
from Util import Index
from NegativeSampling.WrapperNegativeSampler import WrapperNegativeSampler
import constants
from pykeen.evaluation import RankBasedEvaluator
import torch
from torch.optim import Adam

# import dask.dataframe as dd

# import pandas as pd

# file_pattern = "/Users/zafarsaeed/Uniba Italy/Research/Source Code/code/My Test Code/data/index*.json"

# index_file1 = json.load(open("./data/index0.json"))
# df1 = pd.DataFrame(index_file1.items())


# index_file2 = json.load(open("./data/index1.json"))
# df2 = pd.DataFrame(index_file1.items())
# # df = dd.read_json("./data/index0.json")
# frames = [df1, df2]
# combined_df = pd.concat(frames)

# combined_dict = {}

# for key in set(index_file1.keys()).union(index_file2.keys()):
#     # Concatenate corresponding values if key is present in both dictionaries
#     combined_dict[key] = index_file1.get(key, []) + index_file2.get(key, [])

# Load train, validation, and test datasets from text files
# dataset_path = "/Users/zafarsaeed/Uniba Italy/Research/Source Code/Datasets/FB15k-237"
# data_path = "/Users/zafarsaeed/Uniba Italy/Research/Source Code/code/My Test Code/data"
train_file = constants.data_path + "/train"
valid_file = constants.data_path + "/dev"
test_file = constants.data_path + "/test"

# NEED TO CHANGE THE FOLLOWING WITH CONFIGURATION FILE
#config = config = json.load(open(os.path.join(data_path, "{}".format(exp_name), "config.json".format(exp_name))))

# Read the triples from the text files
train_triples = TriplesFactory.from_path(train_file)
validation_triples = TriplesFactory.from_path(valid_file)
test_triples = TriplesFactory.from_path(test_file)

index = Index()

# Specify the wildcard pattern to load files with the same name but different numbers
# file_pattern = 'data_*.json'

# # Use dd.read_json() to load files matching the pattern into a Dask DataFrame
# df = dd.read_json(file_pattern)

# #ZAFAR: Serializing in pickle and save/load
# # Pykeen library is calling CustomeNegativeSample internally and expect it to have a specific signature of the constructor, there, I am unable to modify/customize the parameter list
# # Therefore, I am storing the index on disk, and loading it again inside the CustomeNegativeSampler. It is ugly, but I don't have a solution at this point
# if not index.load_index(constants.data_path):
#     index.ent_index = train_triples.entity_to_id
#     index.rel_index = train_triples.relation_to_id
#     index.save_index(constants.data_path)

if os.path.exists(os.path.join(constants.data_path,"experiment_specs","test_exp_config.json")):
    main_config = json.load(open(os.path.join(constants.data_path,"experiment_specs","test_exp_config.json")))
    if os.path.exists(os.path.join(constants.data_path,main_config["negative_sample_path"],"config.json")):
        constants.experiment_config = json.load(open(os.path.join(constants.data_path,main_config["negative_sample_path"],"config.json")))
    else:
        raise Exception("Experiment configuration file does not exist.")    
else:
    raise Exception("Main configuration file does not exist.")


# pipeline parameters from Configuration
_negative_sampler_kwargs = {
    "mapped_triples": train_triples.mapped_triples,
    "num_negs_per_pos": constants.experiment_config.get("num_negs",1)
}

_training_kwargs = {
        "num_epochs":constants.experiment_config.get("num_epochs",100),
        "batch_size":constants.experiment_config.get("batch_size",500),
        "optimizer": "Adam",
        "negative_sampler_kwargs": _negative_sampler_kwargs
}

_model_kwargs = {
    "embedding_dim":constants.experiment_config.get("ent_dim",500)
}

_evaluation_kwargs = {
    'filter_neg_triples': False,
    'evaluator': 'rankbasedevaluator',  # Specify the evaluator
    'evaluator_kwargs': {
        'at': [1, 3, 10],  # Ranks to compute
    },
}

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the pipeline
results = pipeline(
    training=train_triples,
    testing=test_triples,
    validation=validation_triples,
    model='TransE',
    # model_kwargs=_model_kwargs,
    model_kwargs= dict(embedding_dim= constants.experiment_config.get("ent_dim",500)),
    #negative_sampler='Bernoulli',
    negative_sampler=WrapperNegativeSampler(mapped_triples=train_triples.mapped_triples, num_negs_per_pos=constants.experiment_config.get("num_negs",1)),
    # negative_sampler=WrapperNegativeSampler(mapped_triples=_negative_sampler_kwargs["mapped_triples"],num_negs_per_pos=_negative_sampler_kwargs["num_negs_per_pos"]),
    # negative_sampler=WrapperNegativeSampler,
    # negative_sampler_kwargs=_negative_sampler_kwargs,
    # training_kwargs=_training_kwargs,
    training_kwargs = dict(num_epochs=constants.experiment_config.get("num_epochs",100),batch_size=constants.experiment_config.get("batch_size",500)),
    
    # evaluation=dict(
    #     eval_frequencies=1, 
    #     evaluator=RankBasedEvaluator,
    #     evaluation_kwargs=dict(
    #         k=[1, 3, 5, 10],  # Set the values of k for hit metrics
    #         additional_metrics=['mean_reciprocal_rank']  # Include mean reciprocal rank
    #     )
    # ) #eval_frequencies is set to an integer value n, evaluation will be performed every n epochs during training. This allows you to monitor the performance of the model periodically throughout the training process.
    optimizer=Adam,  # Use Adam optimizer
    optimizer_kwargs=dict(lr=constants.experiment_config.get("lr",0.1)),
    random_seed=1234,
    device=_device
    # device='cuda',  # Change to 'cuda' if you have GPU available
)

model = results.trained_model

evaluator = RankBasedEvaluator()

metrics = evaluator.evaluate(model=results.trained_model, mapped_triples=test_triples.mapped_triples)

# Print the metrics
print(f"Hits@1: {metrics.get_metric('hits@1')}")
print(f"Hits@3: {metrics.get_metric('hits@3')}")
print(f"Hits@5: {metrics.get_metric('hits@5')}")
print(f"Hits@10: {metrics.get_metric('hits@10')}")
print(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")
# Print the evaluation results
print(results)