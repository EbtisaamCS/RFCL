from torch import Tensor
from typing import NewType, Dict, List, Tuple
from enum import Enum

# New Types
Errors = NewType("Errors", Tensor)
IdRoundPair = NewType("IdRoundPair", Tuple[int, int])
BlockedLocations = NewType("BlockedLocations", Dict[str, List[IdRoundPair]])



# Type Aliases
MaliciousList = List[int]
FaultyList = List[int]
AttackName = str
AttacksType = List[Tuple[FaultyList, MaliciousList, AttackName]]






class PersonalisationMethod(Enum):
    """
    Enums for deciding which personalisation method that is wanted for FedPADRC
    """

    SELECTIVE = "Selective"  # The default
   