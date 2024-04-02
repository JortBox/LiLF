import sys, os

lilf_path = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.append(lilf_path)

try:
    from .pipeline_utils import *
    from .peeling_alt import peel
    from .calibration import SelfCalibration

    import LiLF_lib.lib_util as util
    import LiLF_lib.lib_log as log
except:
    pass