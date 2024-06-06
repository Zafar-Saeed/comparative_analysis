
import os
import ujson
import time
import torch
import math
from random import randrange
import numpy as np
import random
import pickle
import constants

# IMPORTS FROM KGE-RL
from negateive_samplers.kge_rl.data_loader import Index

pipeline_call = 0

class CorruptSampler(object):
    def __init__(self,config: dict, index: Index, negative_samples_path, filtered=True):
        
        global pipeline_call
        self.config = config
        self.index = index
        self.negative_samples_path = negative_samples_path
        self.filtered = filtered  # this parameter is for corrupting heads and then tails in the triples

        start_time = time.time()
        #print("Loading pre-generated negative samples...")
        pipeline_call += 1
        self.read_falg = False
        if pipeline_call > 0: #ZAFAR: hardfix, try to find an alternative because pipeline is call this function three times, and momeory is exhusted everytime loading all negative samples from file
            if os.path.exists(os.path.join(
                self.negative_samples_path, 
                "".join([constants.negative_sample_file_prefix, "_FULL.pkl"]))):
                #self.negative_sample_index = ujson.load(open(os.path.join(self.negative_samples_path, "tiples_index_dictionary.json")))
                print("Loading all pre-generated negative samples...")
                self.negative_sample_index = pickle.load(open(os.path.join(negative_samples_path,"tiples_index_dictionary_epoch_FULL.pkl"), 'rb'))
                print("Loading completed...")
                self.read_falg = True
            else:
                raise Exception("Index file for pre-generated negative samples does not exist.")
        else:
            print("Pipeline Call: ", pipeline_call, ", skipping the loading of negative sample index")

        
        end_time = time.time()
        mins = int(end_time - start_time)/60
        secs = int(end_time - start_time)%60

        print("Total time taken to load negative sample index: {0} mins and {1} secs".format(mins,secs))

    def get_typed(self):
        typed = dict()
        for ex in self._triples:
            ents = typed.get(ex.r,tuple([set(),set()]))
            ents[0].add(ex.s)
            ents[1].add(ex.t)
            typed[ex.r] = ents
        #assert len(typed.keys())==constants.fb15k_rels
        return typed

    def get_candidates(self,ex,is_target):
        return self._typed_entities[ex.r][1] if is_target else self._typed_entities[ex.r][0]
    
    def get_all_negatives(self, source: int, relation: int, target: int):
        list_all_negatives = self.negative_sample_index["{} {} {}".format(source,relation,target)]
        # heads = [negative[0] for negative in list_all_negatives]
        # relations = [negative[1] for negative in list_all_negatives]
        # tails = [negative[2] for negative in list_all_negatives]
        
        # return heads,relations,tails
        if len(list_all_negatives) == 0:
            raise Exception("Negative sample set does not exist, something went wrong")
        
        return list_all_negatives
    
    def get_negatives(self, num_of_samples_to_corrupt: int, negative_candidates: list, index: int) -> list:
        negative_selection_indices = list(np.arange(len(negative_candidates)))

        #ZAFAR IMPORTANT: here if negative_candidates < num_of_samples_to_corrupt, then following code may cause issues, recheck this
        if len(negative_candidates) < num_of_samples_to_corrupt:
            raise Exception("Number of possible candidates are less than num_of_samples_to_corrupt")
        elif len(negative_candidates) == num_of_samples_to_corrupt:
            return negative_selection_indices
        else: # I AM HERE, NEED TO CHECK THE FOLLOWING CODE, ABOVE CODE IS CHECKED
            # while len(negative_selection_indices) < num_of_samples_to_corrupt:
            #     index = randrange(len(negative_candidates)) #ZAFAR: it is a hard fix, because python random.randint() has unpredictable behavior, the end range is not inclusive
            #     negative_selection_indices.add(index if index < len(negative_candidates) else index-1) # hard fix forcing the index is not out of bound, however, else case is less likely to execute
            random.shuffle(negative_selection_indices)
            negative_selection_indices = negative_selection_indices[:num_of_samples_to_corrupt]

        return negative_selection_indices
    
    def replace_for_corruption(self, positive_batch: torch.LongTensor, negative_batch: torch.LongTensor, num_negs_per_pos: int, total_num_negatives: int, corruption_indices: list) -> torch.LongTensor:
        
        if self.read_falg: #hard fix, because pipeline is uneccessary calling before running the epoch

            split_idx = int(math.ceil(num_negs_per_pos / len(corruption_indices))) # Zafar: calculating the split for equally corrupting Heads, relations, and tails


            for start in range(0, total_num_negatives, num_negs_per_pos):
                # stop = min(start + num_negs_per_pos, total_num_negatives)
                stop = start + num_negs_per_pos

                # negative_batch[slice(start,stop),0][0:5] -> this will give first five heads 
                # negative_batch[slice(start,stop),0][5:10] -> this will give last five heads 
                # negative_batch[slice(start,stop),1][0:5] -> this will give first five relationships
                # negative_batch[slice(start,stop)][0] -> this will give you the first row of that slice
                # negative_batch[slice(start,stop)][0,0] -> this will give you the first element (head) of first row of that slice
                # negative_batch[slice(start,stop)][0].numpy() -> this will convert tensor into numpy array
                # negative_batch[slice(start,stop)][0:5] -> this will give first five rows in tensor
                chunk_to_corrupt = negative_batch[slice(start,stop)]
                postive_triple = chunk_to_corrupt[0] # all of the triples in chunk is redundant, because it contains positive triple to be corrupted 'num_negs_per_pos' times
                
                #(triple[0],triple[1],triple[2]) => (h,r,t)
                # (heads,relations,tails) => (triple[0],triple[1],triple[2]) # getting the list of possible negatives for 'triple'
                all_negative_candidates = self.get_all_negatives(postive_triple[0],postive_triple[1],postive_triple[2]) # getting the list of possible negatives for 'triple'
                
                chunk_start = chunk_stop = 0
                
                #ZAFAR: Test for [1,2,5,11] num_negs_per_pos to check the handling of correct splits

                for index in corruption_indices:
                    filter_negative_candidates = [negative_triple for negative_triple in all_negative_candidates if negative_triple[index] != postive_triple[index]]
                    
                    chunk_start = chunk_stop
                    chunk_stop = min(chunk_start + split_idx, len(chunk_to_corrupt))
                    num_of_samples_to_corrupt = chunk_stop-chunk_start
                    negative_sample_selection_indices = self.get_negatives(num_of_samples_to_corrupt=num_of_samples_to_corrupt, negative_candidates=filter_negative_candidates, index=index)

                    sub_chunk_to_corrupt = chunk_to_corrupt[chunk_start:chunk_stop]

                    for ind, triple in enumerate(sub_chunk_to_corrupt):
                        triple[index] = filter_negative_candidates[negative_sample_selection_indices[ind]][index]
                
                #index variable is replaced with 0 temporarly for testing
                # replacement = torch.randint(
                #     high=self.num_relations if 0 == 1 else self.num_entities - 1,
                #     size=(stop - start,),
                #     device=negative_batch.device,
                # )
                # replacement += (replacement >= negative_batch[slice(start, stop), 0]).long()
                # negative_batch[slice(start, stop), 0] = replacement

        return negative_batch
        

