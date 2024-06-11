# IMPORTS FROM KGE-RL
from negateive_samplers.kge_rl.data_loader import Index 

class SharedIndex:

    _instance = None

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.index = Index()
            self._initialized = True

    @classmethod
    def get_instance(cls, value=None):
        if cls._instance is None:
            cls._instance = SharedIndex()
        return cls._instance
