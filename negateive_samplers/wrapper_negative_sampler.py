from pykeen.sampling import NegativeSampler
import torch
import random
from pykeen.typing import MappedTriples 
from pykeen.sampling.basic_negative_sampler import random_replacement_

import torch

from pykeen.sampling.negative_sampler import NegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, MappedTriples

from pykeen.typing import Target
#from pykeen.typing import Collection, Optional
from pykeen.typing import Collection
import math
from typing import Optional
from pykeen.constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX

# IMPORTS FROM KGE-RL
from negateive_samplers.kge_rl.data_loader import Index
from negateive_samplers.kge_rl.data_loader import Path
from negateive_samplers.kge_rl import negative_sampling
from negateive_samplers.corrupt_sampler import CorruptSampler
from negateive_samplers.kge_rl import util


import sys
import math


def replacement_for_corruption(data_index: Index, positive_batch: torch.LongTensor,batch: torch.LongTensor, negative_sampler: object, index: int, selection: slice, size: int, max_index: int) -> None:
# def replacement_for_corruption(positive_batch: torch.LongTensor,batch: torch.LongTensor, negative_sampler: object, corruption_indices: list, selection: slice, size: int) -> None:
    """
    Replace a column of a batch of indices by random indices.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """
    # At least make sure to not replace the triples by the original value
    # To make sure we don't replace the {head, relation, tail} by the
    # original value we shift all values greater or equal than the original value by one up
    # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]

    #ZAFAR: Before the replacement, batch parameter contains the positive triples that needs to be corrupted
    #ZAFAR: use the batch to track the negative samples from negative_sample_path files then replace the tiples accordingly
    # negative_sampler = data_index.create_negative_sample_index(dir_name=negative_sample_path)
    
    if isinstance(negative_sampler, CorruptSampler):
        negative_samples = negative_sampler

    heads,relations,tails = negative_samples.get_negatives(187,1040,14413)

    print("Heads: ", heads)
    print("Relations: ", relations)
    print("Tails: ", tails)
    
    
    replacement = torch.randint(
        high=max_index - 1,
        size=(size,),
        device=batch.device,
    )
    replacement += (replacement >= batch[selection, index]).long()
    batch[selection, index] = replacement

class WrapperNegativeSampler(NegativeSampler):
    r"""A basic negative sampler.

    This negative sampler that corrupts positive triples $(h,r,t) \in \mathcal{K}$ by replacing either $h$, $r$ or $t$
    based on the chosen corruption scheme. The corruption scheme can contain $h$, $r$ and $t$ or any subset of these.

    Steps:

    1. Randomly (uniformly) determine whether $h$, $r$ or $t$ shall be corrupted for a positive triple
       $(h,r,t) \in \mathcal{K}$.
    2. Randomly (uniformly) sample an entity $e \in \mathcal{E}$ or relation $r' \in \mathcal{R}$ for selection to
       corrupt the triple.

       - If $h$ was selected before, the corrupted triple is $(e,r,t)$
       - If $r$ was selected before, the corrupted triple is $(h,r',t)$
       - If $t$ was selected before, the corrupted triple is $(h,r,e)$
    3. If ``filtered`` is set to ``True``, all proposed corrupted triples that also exist as
       actual positive triples $(h,r,t) \in \mathcal{K}$ will be removed.
    """

    def __init__(
        self,
        *,
        corruption_scheme: Optional[Collection[Target]] = None,
        config,
        data_for_ns,
        results_dir,
        **kwargs,
    ) -> None:
        """Initialize the basic negative sampler with the given entities.

        :param corruption_scheme:
            What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        # if "triples" in kwargs:
        #     self.triples = kwargs["triples"]
        #     kwargs["mapped_triples"] = kwargs["triples"].mapped_triples
        #     kwargs.pop("triples")

        
        super().__init__(**kwargs)
        
        self.config = config

        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        # Set the indices
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        
        #Zafar: Hard fix, because PyKeen devides the num_negs_per_pos for all sides
        #Zafar: in kge-rl they generate equal nun_negs_per_pos for each side
        #Zafar: according to the existing code, I am hard fixing it here
        self.num_negs_per_pos = self.config.get("num_negs",1) * len(self._corruption_indices)

        self.sampler = self.get_neg_sampler(data_for_ns, results_dir)
    
    #Zafar: new function
    def get_neg_sampler(self, triples, results_dir):
        
        if self.config['neg_sampler'] == 'random':
            return negative_sampling.Random_Sampler(triples,self.config["num_negs"])
        elif self.config['neg_sampler'] == 'corrupt':
            return negative_sampling.Corrupt_Sampler(triples,self.config["num_negs"])
        elif self.config['neg_sampler'] == 'typed':
            return negative_sampling.Typed_Sampler(triples,self.config["num_negs"],results_dir)
        elif self.config['neg_sampler'] == 'relational':
            return negative_sampling.Relational_Sampler(triples,self.config["num_negs"])
        elif self.config['neg_sampler'] == 'nn':
            return negative_sampling.NN_Sampler(triples,self.config["num_negs"])
        elif self.config['neg_sampler'] == 'adversarial':
            return negative_sampling.Adversarial_Sampler(triples, self.config["num_negs"])
        elif self.config['neg_sampler'] == 'rl':
            return negative_sampling.Policy_Sampler(triples, self.config["num_negs"])
        else:
            raise NotImplementedError("Neg. Sampler {} not implemented".format(self.config['neg_sampler']))


        
    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]
        
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        # Zafar: self._corruption_indices shows the number of sides to be corrupted. Head being 0, Relationship beeing 1, and Trail being 2
            
        # if isinstance(self.sampler, CorruptSampler):
        #         negative_sampler = self.sampler
        #         negative_batch = negative_sampler.replace_for_corruption(
        #             positive_batch=positive_batch,
        #             negative_batch=negative_batch,
        #             num_negs_per_pos=self.num_negs_per_pos,
        #             total_num_negatives=total_num_negatives,
        #             corruption_indices=self._corruption_indices)

        if isinstance(self.sampler, negative_sampling.Corrupt_Sampler):
                negative_sampler = self.sampler
                # negative_batch = negative_sampler.replace_for_corruption(
                #     positive_batch=positive_batch,
                #     negative_batch=negative_batch,
                #     num_negs_per_pos=self.num_negs_per_pos,
                #     total_num_negatives=total_num_negatives,
                #     corruption_indices=self._corruption_indices)

                # add code here to convert the batch structure that negative_sampling.Corrupt_Sampler expects
                # the existing code takes batch as list(Path)
                negative_sampler_postive_batch = [Path(*(path.tolist())) for path in positive_batch]

                
                negs_targets = negative_sampler.batch_sample(negative_sampler_postive_batch, is_target=True)
                #negative_batch_targets = util.get_triples_lists(negative_sampler_postive_batch, negs_targets, True, volatile=False)
                negs_sources = negative_sampler.batch_sample(negative_sampler_postive_batch, is_target=False)
                # negative_batch_sources = util.get_triples_lists(negative_sampler_postive_batch, negs, False, volatile=False)

                split_index = math.ceil(self.num_negs_per_pos/2) #no need of using ceil function, as self.num_negs_per_pos was already multiply by 2 in constructor. when that code is removed from constructor then this line will make sense
                
                #safty check: both sides (sources, targers) must have equal number of negatives
                if len(negs_sources) != len(negs_targets):
                    raise Exception("negs_sources is not equal to negs_targets")
                else:
                    num_batch_samples = len(positive_batch)

                start = end = 0

                if num_batch_samples < 500:
                    dummy_break = 0

                for index in range(0,num_batch_samples):
                    start = index * split_index * 2 # multiplying with two because both sides are corrupted in single loop, therefore, each iteration would jump index twoice
                    end = start + split_index
                    selection = slice(start,end)
                
                    #corrupting target side
                    negative_batch[selection,2] = torch.tensor(negs_targets[index], device=self.config["device"])
            
                    start = end
                    end = start + split_index
                    selection = slice(start,end)
                    #corrupting for source side
                    negative_batch[selection,0] = torch.tensor(negs_sources[index], device=self.config["device"])
            

                # for index,target_list in enumerate(negs_targets):
                #     start = index*split_index
                #     end = start+split_index
                #     selection = slice(start,end)
                
                #     negative_batch[selection,2] = torch.tensor(target_list, device=self.config["device"])
                
                #     # start = end
                #     # end = self.num_negs_per_pos
                #     # selection = slice(start,end)
                #     selection.start = end
                #     selection.st = self.num_negs_per_pos
                    
                
                # negative_batch = negative_sampler.replace_for_corruption(
                #     positive_batch=positive_batch,
                #     negative_batch=negative_batch,
                #     num_negs_per_pos=self.num_negs_per_pos,
                #     total_num_negatives=total_num_negatives,
                #     corruption_indices=self._corruption_indices)

        

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def convert_tensor_to_list():
        path_list = [Path(int(s), int(r), int(t)) for s, r, t in tensor]
