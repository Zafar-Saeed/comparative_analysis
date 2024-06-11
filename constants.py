# __author__=  'Bhushan Kotnis'

#Index file names
entity_ind = 'entities.cpkl'
rel_ind = 'relations.cpkl'
negative_sample_file_prefix = "tiples_index_dictionary_epoch"

'''SGD Batch Size'''
batch_size = 500
test_batch_size = 10

'''Negatives'''
num_train_negs = 10
num_dev_negs = 1000
num_test_negs = float('inf')


'''Report and Save model'''
report_steps = 200


'''Early Stopping'''
early_stop_counter = 5
patience = 15
num_epochs=100

'''Dataset details'''
fb15k_rels = 1345
fb15k_ents = 14951

wn_rels = 18
wn_ents = 40943

fb13_ents=75043
fb13_rels=13
#cat_file='/home/mitarb/kotnis/Code/kge-rl/entity_cat.cpkl'

#ZAFAR
# /Users/zafarsaeed/Uniba Italy/Research/Source Code/code/comparative_analysis/negateive_samplers/kge_rl/entity_cat.cpkl
cat_file='entity_cat.cpkl'
data_path = "./data"
experiment_config = dict()
