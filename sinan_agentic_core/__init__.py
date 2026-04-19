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

from .core import BaseAgentRunner, ToolErrorRecovery, ToolErrorRecoveryHooks, TurnBudget, TurnBudgetHooks
from .instructions import InstructionBuilder
from .session import AgentSession, ConversationHistory, SQLiteSessionStore
from .models.context import AgentContext
from .models.outputs import ToolOutput, ChatResponse
from .registry import (
    AgentCatalog,
    AgentYamlEntry,
    load_agent_catalog,
    AgentDefinition,
    AgentRegistry,
    get_agent_registry,
    register_agent,
    create_agent_from_registry,
    ToolCatalog,
    ToolYamlEntry,
    load_tool_catalog,
    ToolDefinition,
    ToolRegistry,
    get_tool_registry,
    register_tool,
)
from .services import (
    StreamingRunHooks,
    chat,
    chat_with_hooks,
    chat_streamed,
)

from .orchestrator import AgentOrchestrator
from .utils import tool_error, tool_response, unwrap_context

# MCP support (optional — requires 'sinan_agentic_core[mcp]')
# Lazy import to avoid requiring FastMCP for non-MCP users.
# Usage: from sinan_agentic_core.mcp import build_mcp_server, MCPContextFactory

__all__ = [
    # Core
    "BaseAgentRunner",
    "ToolErrorRecovery",
    "ToolErrorRecoveryHooks",
    "TurnBudget",
    "TurnBudgetHooks",
    # Instructions
    "InstructionBuilder",
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
