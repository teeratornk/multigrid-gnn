"""A fixed-weight algebraic multigrid V-cycle executed as message-passing layers."""
from .cases import build_case
from .hierarchy import build_hierarchy
from .layers import MessagePassingOperator
from .pcg import pcg
from .vcycle import v_cycle

__all__ = ["build_case", "build_hierarchy", "MessagePassingOperator", "pcg", "v_cycle"]
__version__ = "1.2.2"
