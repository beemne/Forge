import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger("forge.agents")

class AgentMessage(BaseModel):
    agent: str
    action: str
    capability: Optional[str] = None
    reason: str = ""
    payload: Dict[str, Any] = {}

class BaseAgent:
    def __init__(self, name: str, default_capability: str):
        self.name = name
        self.default_capability = default_capability

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="orchestrator", default_capability="recon")

class ReconAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="recon", default_capability="network_scanning")

class WebAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="web", default_capability="directory_enumeration")

class ForensicsAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="forensics", default_capability="file_analysis")

class CryptoAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="crypto", default_capability="general_reasoning")

class PwnAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="pwn", default_capability="reverse_engineering")

class RevAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="rev", default_capability="reverse_engineering")

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "orchestrator": OrchestratorAgent(),
            "recon": ReconAgent(),
            "web": WebAgent(),
            "forensics": ForensicsAgent(),
            "crypto": CryptoAgent(),
            "pwn": PwnAgent(),
            "rev": RevAgent()
        }

    def get_agent(self, name: str) -> BaseAgent:
        return self.agents.get(name, self.agents["orchestrator"])

agent_manager = AgentManager()
