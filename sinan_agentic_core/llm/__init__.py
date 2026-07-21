"""External LLM provider support — public OpenAI and Azure OpenAI.

Quick start::

    from sinan_agentic_core.llm import load_llm_provider_config, configure_llm_provider

    cfg = load_llm_provider_config("config.yaml")  # reads the top-level 'llm:' key
    configure_llm_provider(cfg)
"""

from .config import (
    AzureOpenAIProviderConfig,
    LLMProviderConfig,
    OpenAIProviderConfig,
    parse_llm_provider_config,
)
from .factory import configure_llm_provider, resolve_openai_client
from .yaml_loader import load_llm_provider_config

__all__ = [
    "AzureOpenAIProviderConfig",
    "LLMProviderConfig",
    "OpenAIProviderConfig",
    "configure_llm_provider",
    "load_llm_provider_config",
    "parse_llm_provider_config",
    "resolve_openai_client",
]
