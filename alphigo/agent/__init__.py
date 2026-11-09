from .base import Agent
from .naive import RandomBot
from .registry import create_agent, list_agents

try:
    from .policy_agent import PolicyAgent, load_policy_agent
    from .q_agent import QAgent, load_q_agent
except ImportError:  # TensorFlow not installed
    pass
