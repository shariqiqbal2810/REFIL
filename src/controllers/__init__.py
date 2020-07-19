REGISTRY = {}

from .basic_controller import BasicMAC
from .entity_controller import EntityMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["entity_mac"] = EntityMAC
