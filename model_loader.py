from pykeen.models import *
from pykeen.triples import TriplesFactory
from pykeen.regularizers import Regularizer
from pykeen.typing import HintOrType
from pykeen.losses import Loss
import torch

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
        if model_name.lower() == "transe":
            model = TransE(
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
    else:
        raise Exception("Model cannot be initialized...")
    return model


def load_pre_trained_model(file_path: str):

    model = Model.load_state(file_path)
    return model
