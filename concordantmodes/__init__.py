from importlib.metadata import PackageNotFoundError, version as _pkg_version

from concordantmodes.algorithm import Algorithm
from concordantmodes.cma import ConcordantModes
from concordantmodes.directory_tree import DirectoryTree
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.force_constant import ForceConstant
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.g_read import GrRead
from concordantmodes.int2cart import Int2Cart
from concordantmodes.molden_writer import MoldenWriter
from concordantmodes.reap import Reap
from concordantmodes.rmsd import RMSD
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp

from concordantmodes.options import Options
from concordantmodes.errors import CMAError, CMAWaiting, OptionsError

try:
    __version__ = _pkg_version("concordantmodes")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
