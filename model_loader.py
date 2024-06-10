from pykeen.models import *
from pykeen.triples import TriplesFactory
from pykeen.regularizers import Regularizer
from pykeen.typing import HintOrType
from pykeen.losses import Loss
import torch
# from torch.nn.modules.activation import Tanh

def load_model(
        model_name: str | None,
        train_tripple_factory: TriplesFactory,
        regularizer: HintOrType[Regularizer],
        loss: HintOrType[Loss],
        embedding_dimensions: int,
        decive: torch.device,
        random_seed: int = 1234
        ):

    if model_name is not None:
        print("Building {} model...".format(model_name))
        if model_name.lower() == "transe":
            model = TransE(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "transh":
            model = TransH(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "transr":
            model = TransR(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "transd":
            model = TransD(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "distmult":
            model = DistMult(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "complex":
            model = ComplEx(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "simple":
            model = SimplE(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "rotate":
            model = RotatE(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "rescal":
            model = RESCAL(
                triples_factory=train_tripple_factory,
                regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "tucker":
            model = TuckER(
                triples_factory=train_tripple_factory,
                #regularizer=regularizer,  #TuckER does not take regularizer as parameter
                #following are the defult dropout values
                dropout_0=0.3,  # Dropout rate for entity embeddings
                dropout_1=0.4,  # Dropout rate for the first mode of the tensor
                dropout_2=0.5,  # Dropout rate for the second mode of the tensor
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "quate":
            model = QuatE(
                triples_factory=train_tripple_factory,
                entity_regularizer=regularizer,
                relation_regularizer=regularizer,
                #regularizer=regularizer,
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "hole":
            model = HolE(
                triples_factory=train_tripple_factory,
                #regularizer=regularizer, #HolE does not take regularizer as parameter
                # entity_constrainer_default_kwargs= {'dim': -1, 'maxnorm': 1.0, 'p': 2} # these are the default parameters and handles L2 regularizer 
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
        elif model_name.lower() == "boxe":
            model = BoxE(
                triples_factory=train_tripple_factory,
                #regularizer=regularizer, #BoxE does not take regularizer as parameter
                # tanh_map (bool) – Whether to use tanh mapping after BoxE computation (defaults to true).
                # The hyperbolic tangent mapping restricts the embedding space to the range [-1, 1], 
                # and thus this map implicitly regularizes the space to prevent loss reduction by growing boxes arbitrarily large.
                random_seed=random_seed,
                loss=loss,
                embedding_dim=embedding_dimensions
                ).to(decive)
    else:
        raise Exception("Model cannot be initialized...")
    return model


def load_pre_trained_model(file_path: str):

    model = Model.load_state(file_path)
    return model
