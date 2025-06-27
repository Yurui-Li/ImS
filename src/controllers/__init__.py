REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ims_controller import ImSMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ims"] = ImSMAC
