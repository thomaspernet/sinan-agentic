"""Sinan - Generic agent orchestration framework.

This package provides a reusable framework for building multi-agent systems
using the OpenAI Agents SDK with code orchestration.

Core Components:
- AgentSession: Manages conversation history
- AgentContext: Stores workflow state and data
- AgentRegistry: Declarative agent definitions
- ToolRegistry: Declarative tool definitions
- AgentOrchestrator: Workflow orchestration logic

Quick Start:
    from sinan_agentic_core import (
        AgentSession,
        AgentContext,
        AgentOrchestrator,
        register_agent,
        register_tool,
        AgentDefinition
    )

    # Define your agent
    my_agent = AgentDefinition(
        name="my_agent",
        description="Does something",
        instructions="You are a helpful assistant...",
        tools=["my_tool"]
    )
    register_agent(my_agent)

    # Run orchestrator
    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(
        user_query="Do something",
        context_data={"database_connector": db}
    )
"""

from .core import (
    BaseAgentRunner,
    Capability,
    ModelRetryConfig,
    RetryBackoffConfig,
    RetryTrigger,
    ToolErrorRecovery,
    ToolOutputTrimConfig,
    ToolTracer,
    TurnBudget,
    recover_invalid_final_output,
)
from .instructions import InstructionBuilder
from .llm import (
    AzureOpenAIProviderConfig,
    LLMProviderConfig,
    OpenAIProviderConfig,
    configure_llm_provider,
    load_llm_provider_config,
    parse_llm_provider_config,
)
from .models.context import AgentContext
from .models.outputs import ChatResponse, ToolOutput
from .orchestrator import AgentOrchestrator
from .registry import (
    AgentCatalog,
    AgentDefinition,
    AgentRegistry,
    AgentYamlEntry,
    CapabilityFactory,
    CapabilityNotFoundError,
    CapabilityRef,
    CapabilityRegistry,
    GuardrailCategory,
    GuardrailDefinition,
    GuardrailRegistry,
    ResolvedGuardrails,
    ToolCatalog,
    ToolDefinition,
    ToolRegistry,
    ToolYamlEntry,
    create_agent_from_registry,
    get_agent_registry,
    get_capability_registry,
    get_guardrail_registry,
    get_tool_registry,
    load_agent_catalog,
    load_tool_catalog,
    register_agent,
    register_capability,
    register_guardrail,
    register_tool,
)
from .services import (
    StreamingRunHooks,
    chat,
    chat_streamed,
    chat_with_hooks,
)
from .session import AgentSession, ConversationHistory, SQLiteSessionStore
from .utils import tool_error, tool_response, unwrap_context

# MCP support (optional — requires 'sinan_agentic_core[mcp]')
# Lazy import to avoid requiring FastMCP for non-MCP users.
# Usage: from sinan_agentic_core.mcp import build_mcp_server, MCPContextFactory

__all__ = [
    # Core
    "BaseAgentRunner",
    "Capability",
    "ModelRetryConfig",
    "recover_invalid_final_output",
    "RetryBackoffConfig",
    "RetryTrigger",
    "ToolErrorRecovery",
    "ToolOutputTrimConfig",
    "ToolTracer",
    "TurnBudget",
    # Instructions
    "InstructionBuilder",
    # LLM providers
    "AzureOpenAIProviderConfig",
    "LLMProviderConfig",
    "OpenAIProviderConfig",
    "configure_llm_provider",
    "load_llm_provider_config",
    "parse_llm_provider_config",
    # Session
    "AgentSession",
    "ConversationHistory",
    "SQLiteSessionStore",
    # Models
    "AgentContext",
    "ToolOutput",
    "ChatResponse",
    # Registry
    "AgentCatalog",
    "AgentYamlEntry",
    "CapabilityFactory",
    "CapabilityNotFoundError",
    "CapabilityRef",
    "CapabilityRegistry",
    "get_capability_registry",
    "register_capability",
    "load_agent_catalog",
    "AgentDefinition",
    "AgentRegistry",
    "get_agent_registry",
    "register_agent",
    "create_agent_from_registry",
    "ToolCatalog",
    "ToolYamlEntry",
    "load_tool_catalog",
    "ToolDefinition",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    "GuardrailCategory",
    "GuardrailDefinition",
    "GuardrailRegistry",
    "ResolvedGuardrails",
    "get_guardrail_registry",
    "register_guardrail",
    # Services
    "StreamingRunHooks",
    "chat",
    "chat_with_hooks",
    "chat_streamed",
    # Orchestrator
    "AgentOrchestrator",
    # Utils
    "unwrap_context",
    "tool_response",
    "tool_error",
]
