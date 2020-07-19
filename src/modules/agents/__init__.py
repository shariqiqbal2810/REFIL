REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .entity_rnn_agent import ImagineEntityAttentionRNNAgent, EntityAttentionRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["entity_attend_rnn"] = EntityAttentionRNNAgent
REGISTRY["imagine_entity_attend_rnn"] = ImagineEntityAttentionRNNAgent
