from .base import StructuredModel
from .multiclass import M3NMultiClass, M4NMultiClass, CRFMultiClass
from .sequence import M3NChainFactorGraph, M4NChainFactorGraph, CRFChainFactorGraph
from .matching import M3NMatching, M4NMatching

__all__ = ["StructuredModel",
           "M3NMultiClass", "M4NMultiClass", "CRFMultiClass",
           "M3NChainFactorGraph", "M4NChainFactorGraph", "CRFChainFactorGraph", 
           "M3NMatching", "M4NMatching"]
