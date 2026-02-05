"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver



from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
}