import json
import os
import constants
from pykeen.triples import TriplesFactory
from negateive_samplers.wrapper_negative_sampler import WrapperNegativeSampler
import torch
# Pick a model
from pykeen.regularizers import LpRegularizer
from pykeen.models import TransE
# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import LCWAEvaluationLoop, RankBasedEvaluator
from pykeen.optimizers import Adam
from pykeen.losses import MarginRankingLoss
from pykeen.stoppers import EarlyStopper
import copy
import sys
import time
from datetime import datetime

# my files or classes
import util


# IMPORTS FROM KGE-RL
from negateive_samplers.kge_rl import data_loader


def main(exp_name,data_path, is_code_testing=False, dataset_name=None, is_model_test=False):

    # dataset_name = 'nations'
    # util.write_pykeen_dataset(data_path, test_dataset_name)
    # os.makedirs("./data/directory1/directory2",exist_ok=True)
    # sys.exit(0)

    # for running all the experiments, I need to load all the config files in folder experiment_specs and move the following code in saperate function for running experiments in parallel
    # in that function I also need to check if some code has crashed and what files are present the the result_dir. 
    # If the trained model is saved, then I might need to resume for evaluation process only
    if os.path.exists(os.path.join(data_path,"experiment_specs","exp_config.json")):
        config = json.load(open(os.path.join(data_path,"experiment_specs","exp_config.json")))    
    else:
        print("Main configuration file does not exist. Exiting")
        return

    if is_code_testing: # temporary for testing
        if dataset_name == None:
            config["dataset_name"] = "Nations"
            config["batch_size"]=50  
        else:
            config["dataset_name"] = dataset_name
            config["batch_size"]=100  
            
        config["num_negs"]=10
        config["num_epochs"]=50
        util.write_pykeen_dataset(data_path, dataset_name)
          

    
    config["exp_name"] = config["model"]+"_"+str(config["num_negs"])
    config["results_dir"] =  os.path.join(data_path,"Results",config["dataset_name"],config["neg_sampler"])
    #config["data_index_path"] = os.path.join(data_path,"Results",config["dataset_name"]) # for storing entity and relation index in pickle
    config["dataset_path"] = os.path.join(data_path, config["dataset_name"])
    # if os.path.exists(results_dir):
    #     print("{} already exists, Overwriting existing files.\n".format(results_dir))
    #     # return
    # else:
    #     os.makedirs(results_dir)
    os.makedirs(config["results_dir"], exist_ok=True)

    util.dump_json(config, config["results_dir"], "{}_config.json".format(config["exp_name"]))
    
    #Zafar: unecessary step, but I have to load the dataset in structur supported by existing code so that negative sampler can be created
    print("Loading dataset for reusable code..")
    data_set = data_loader.read_dataset(
        config["dataset_path"], 
        config["dataset_path"], # this is result directory where the entity and relation index will be stored in pickle
        dev_mode= not is_model_test,
        max_examples=float('inf')
    )

    #IMPORTANT: need to cross check why train and test datasets are combined.
    # check in the existing code what they are doing
    data_for_negative_sampler = copy.copy(data_set['train'])
    data_for_negative_sampler.extend(data_set['test'])
    
    cuda = torch.cuda.is_available()
    print("CUDA is available: {}".format(cuda))

    train_file = os.path.join(config["dataset_path"], "train")
    valid_file = os.path.join(config["dataset_path"], "dev")
    test_file = os.path.join(config["dataset_path"], "test")

    index = data_loader.Index()

    index.load_index(config["dataset_path"])
   
    from typing import Mapping

    EntityMapping = Mapping[str, int]
    RelationMapping = Mapping[str, int]

    entity_mapping: EntityMapping = index.ent_index
    relation_mapping: RelationMapping = index.rel_index

# region Different ways to load dataset in pykeen

    # from pykeen.datasets import get_dataset
    # from pykeen.datasets.base import PathDataset

    # dataset = PathDataset(
    #      training_path=train_file,
    #     validation_path=valid_file,
    #     testing_path=test_file,
    #     eager=True,
    #     create_inverse_triples=False,
    #     load_triples_kwargs= dict(
    #         entity_to_id=index.ent_index, 
    #         relation_to_id=index.rel_index),
    #     )

    # ent_mapping = Mapping[str, int]
    # rel_mapping = Mapping[str, int]

    #dataset = Dataset.from_path(dataset_path)
    # dataset = get_dataset(
    #     dataset_kwargs= dict(
    #         eager=True, 
    #         entity_to_id=index.ent_index, 
    #         relation_to_id=index.rel_index),
    #     training=train_file,
    #     validation=valid_file,
    #     testing=test_file)
# endregion

    # Read the triples from the text files
    print("Loading training tripples..")
    train_triples = TriplesFactory.from_path(
        path=train_file, 
        entity_to_id=entity_mapping, 
        relation_to_id=relation_mapping)

    print("Loading validation tripples..")
    validation_triples = TriplesFactory.from_path(
        path=valid_file, 
        entity_to_id=entity_mapping, 
        relation_to_id=relation_mapping)

    print("Loading test tripples..")
    test_triples = TriplesFactory.from_path(
        path=test_file, 
        entity_to_id=entity_mapping, 
        relation_to_id=relation_mapping)

# region Pykeen code Beyond Pipeline    
    # train_triples = dataset.training
    # validation_triples = dataset.validation
    # test_triples = dataset.testing


    # # Get a training dataset
    # from pykeen.datasets import Nations
    # dataset = Nations()
    # training_triples_factory = dataset.training

    # # Pick a model
    # from pykeen.models import TransE
    # model = TransE(triples_factory=training_triples_factory)

    # # Pick an optimizer from Torch
    # from torch.optim import Adam
    # optimizer = Adam(params=model.get_grad_params())

    # # Pick a training approach (sLCWA or LCWA)
    # from pykeen.training import SLCWATrainingLoop
    # training_loop = SLCWATrainingLoop(
    #     model=model,
    #     triples_factory=training_triples_factory,
    #     optimizer=optimizer,
    # )

    # # Train like Cristiano Ronaldo
    # _ = training_loop.train(
    #     triples_factory=training_triples_factory,
    #     num_epochs=5,
    #     batch_size=256,
    # )

    # # Pick an evaluator
    # from pykeen.evaluation import RankBasedEvaluator
    # evaluator = RankBasedEvaluator()

    # # Get triples to test
    # mapped_triples = dataset.testing.mapped_triples

    # # Evaluate
    # results = evaluator.evaluate(
    #     model=model,
    #     mapped_triples=mapped_triples,
    #     batch_size=1024,
    #     additional_filter_triples=[
    #         dataset.training.mapped_triples,
    #         dataset.validation.mapped_triples,
    #     ],
    # )
    # print(results)
# endregion 

    # get a dataset
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["device"] = _device

    _regularizer = LpRegularizer(
        p=2, 
        weight=config.get("l2",0.0001),
        )  # L2 regularization with defult weight 0.0001


    _loss = MarginRankingLoss(
        margin=1
        )

    #Following are the part of model
    # regularizer
    #HPO -> hyper parameter optimization --- there are defult settings if this is not explicitly initialized
        # embedding_dimensions: {type: int, low: 16, high: 256}
    _model = TransE(
        triples_factory=train_triples,
        regularizer=_regularizer,
        random_seed=1234,
        loss=_loss,
        embedding_dim=config.get("ent_dim",50)
        # device = _device
        ).to(_device)

    _optimizer_kwargs= dict(lr=config.get("lr",0.01))
    _optimizer = Adam(params=_model.parameters(), **_optimizer_kwargs)

# region Some training loop parameters 
    #following are the parts of training_loop
    # device# device => cuda or cpu
    #HPO -> hyper parameter optimization --- there are defult settings if this is not explicitly initialized
        # num_epoch: {type: int, 'low': 100, 'high': 1000, 'q': 100}
        #batch_size: {type: int, 'low': 4, 'high': 12, 'scale': 'power_two'}
    #optimizer -> Default is Adam with default lr = 0.001
# endregion
    _training_loop = SLCWATrainingLoop(
        model=_model,
        triples_factory=train_triples,
        negative_sampler=WrapperNegativeSampler(
            mapped_triples=train_triples.mapped_triples,
            num_negs_per_pos=config.get("num_negs",1),
            config=config,
            results_dir=config["results_dir"],
            data_for_ns=data_for_negative_sampler
            ),
        optimizer=_optimizer,
        optimizer_kwargs=_optimizer_kwargs,
        # additional_filter_triples=[train_triples.mapped_triples,validation_triples.mapped_triples],
        #device=_device,
    )

    # TypeError: EarlyStopper.__init__() missing 4 required positional arguments: 'model', 'evaluator', 'training_triples_factory', and 'evaluation_triples_factory'
    # early_stoper = EarlyStopper(
    #     frequency=5,
    #     patience=15,
    #     relative_delta=0.0001)



    #IMPORTANT: train functions has so many parameters including EARLY STOPPER
    # Evaluation callbacks can be used for early stoping
    train_start_time = time.time()
    print("***Training Started*** \t Time: {}".format(datetime.now()))
    losses_per_epoch = _training_loop.train(
        triples_factory=train_triples,
        num_epochs=config.get("num_epochs",1),
        batch_size=config.get("batch_size",100),
        clear_optimizer=True,
        # slice_size=5,
        # sub_batch_size=10,
        # num_workers=10,
        # early_stopping=EarlyStopper(
        #     model=_model,
        #     evaluator=RankBasedEvaluator(),
        #     triples_factory=validation_triples,
        #     frequency=5,
        #     patience=15,
        #     relative_delta=0.001
        # )
        # additional_filter_triples=[train_triples.mapped_triples,validation_triples.mapped_triples],
        #regularizer=_regularizer,
        # NEW: validation evaluation callback
        # callbacks="evaluation-loop", 
        callbacks_kwargs=dict(
            prefix="validation",
            factory=validation_triples, # here the evaluation-loop is used for 
            ),
    )
    train_end_time = time.time()
    print("***Training Comlpeted*** \t Time: {}".format(datetime.now()))
    
    total_training_time = "days: {}, hours: {}, minutes: {}, seconds: {}".format(*util.time_difference(train_start_time, train_end_time))
    
    print("Writing trained model to directory: {}".format(config["results_dir"]))
    _model.save_state(os.path.join(config["results_dir"],"{}.model".format(config["exp_name"]))) # _model.load_state() will load the file and continue
    
    util.dump_json(losses_per_epoch, config["results_dir"], "{}_losses.json".format(config["exp_name"]))

# region this callback function code is not in use    
    # Define a list to store training and evaluation results
    training_results = []
    evaluation_results = []

    def callback(epoch, training_loss, evaluation_loss):
        training_results.append(training_loss)
        evaluation_results.append(evaluation_loss)
        print(f"Epoch {epoch}: Training Loss: {training_loss}, Evaluation Loss: {evaluation_loss}")

# endregion

    print("Model size in memory: {}".format(sys.getsizeof(_model)))
    temp = _model.eval()
    print("Model size in memory after cleaning: {}".format(sys.getsizeof(temp)))
    
    # evaluator = RankBasedEvaluator()
    evaluation_loop = LCWAEvaluationLoop(
        model=_model,
        evaluator=RankBasedEvaluator(),
        evaluator_kwargs=dict(
            filtered=True,
            ),
        triples_factory=test_triples,
        additional_filter_triples=[
            train_triples.mapped_triples,
            validation_triples.mapped_triples,
            ],
        #callbacks=callback,
    )

    print("***Evaluation Started (*** \t Time: {}".format(datetime.now))
    eval_start_time = time.time()
    results = evaluation_loop.evaluate()
    eval_end_time = time.time()
    print("***Evaluation Ended (*** \t Time: {}".format(datetime.now))
    total_eval_time = "days: {}, hours: {}, minutes: {}, seconds: {}".format(*util.time_difference(eval_start_time, eval_end_time))

    results_dict = results.to_dict()
    #print(results_dict)

    util.dump_json(results_dict,config["results_dir"],"{}_metrics.json".format(config["exp_name"]))

    # print("***Explicit Evaluation Started (*** \t Time: {}".format(datetime.now))
    # evaluator = RankBasedEvaluator()

    # exp_eval_start_time = time.time()
    # exp_results = evaluator.evaluate(
    #     model=_model,
    #     mapped_triples=test_triples.mapped_triples,
    #     batch_size=500,
    #     additional_filter_triples=[
    #         train_triples.mapped_triples,
    #         validation_triples.mapped_triples,
    #     ],
    # )
    # exp_eval_end_time = time.time()
    # total_exp_eval_time = "days: {}, hours: {}, minutes: {}, seconds: {}".format(*util.time_difference(exp_eval_start_time, exp_eval_end_time))
    # print("***Explicit Evaluation Ended (*** \t Time: {}".format(datetime.now))
    
    my_results_dic = dict(
        hit_at_1=results.get_metric('hits@1'),
        hit_at_3=results.get_metric('hits@3'),
        hit_at_5=results.get_metric('hits@5'),
        hit_at_10=results.get_metric('hits@5'),
        MRR=results.get_metric('mean_reciprocal_rank'),
        # exp_hit_at_1=exp_results.get_metric('hits@1'),
        # exp_hit_at_3=exp_results.get_metric('hits@3'),
        # exp_hit_at_5=exp_results.get_metric('hits@5'),
        # exp_hit_at_10=exp_results.get_metric('hits@5'),
        # exp_MRR=exp_results.get_metric('mean_reciprocal_rank'),
        training_time=total_training_time,
        evaluation_time=total_eval_time,
        # explicit_evaluation_time=total_exp_eval_time,
    )

    util.dump_json(my_results_dic,config["results_dir"],"{}_metrics_hit_mrr.json".format(config["exp_name"]))

    print(f"Hits@1: {results.get_metric('hits@1')}")
    print(f"Hits@3: {results.get_metric('hits@3')}")
    print(f"Hits@5: {results.get_metric('hits@5')}")
    print(f"Hits@10: {results.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {results.get_metric('mean_reciprocal_rank')}")
    # Print the evaluation results
    print(results)


import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name')
    parser.add_argument('data_path')
    parser.add_argument('--test_code', dest='is_code_testing', action='store_true', help='A boolean flag')
    parser.add_argument('test_dataset_name')
    parser.add_argument('--test_model', dest='is_model_testing', action='store_true', help='A boolean flag')
    
    args = parser.parse_args()
    main(args.exp_name,args.data_path, args.is_code_testing, args.test_dataset_name, args.is_model_testing)
