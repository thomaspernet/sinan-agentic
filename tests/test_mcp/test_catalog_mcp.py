"""Tests for MCP extensions to ToolCatalog and AgentCatalog."""

import pytest

from sinan_agentic_core.registry.tool_catalog import ToolCatalog
from sinan_agentic_core.registry.agent_catalog import AgentCatalog


# ---------------------------------------------------------------------------
# ToolCatalog MCP tests
# ---------------------------------------------------------------------------


def test_tool_catalog_get_mcp_tools():
    """get_mcp_tools() returns only tools with mcp.expose: true."""
    catalog = ToolCatalog(raw_tools={
        "discover": {
            "description": "Discover",
            "mcp": {"expose": True},
        },
        "search": {
            "description": "Search",
            "mcp": {"expose": True},
        },
        "think": {
            "description": "Think",
            # No mcp section → not exposed
        },
        "ask_user": {
            "description": "Ask user",
            "mcp": {"expose": False},
        },
    })

    mcp_tools = catalog.get_mcp_tools()
    assert sorted(mcp_tools) == ["discover", "search"]


def test_tool_catalog_get_mcp_tools_empty():
    """get_mcp_tools() returns empty list when no tools have mcp section."""
    catalog = ToolCatalog(raw_tools={
        "think": {"description": "Think"},
    })
    assert catalog.get_mcp_tools() == []


def test_tool_yaml_entry_mcp_field():
    """ToolYamlEntry should include mcp config when present."""
    catalog = ToolCatalog(raw_tools={
        "discover": {
            "description": "Discover",
            "mcp": {
                "expose": True,
                "annotations": {"readOnlyHint": True},
            },
        },
    })

    entry = catalog.get("discover")
    assert entry.mcp is not None
    assert entry.mcp.expose is True
    assert entry.mcp.annotations == {"readOnlyHint": True}


def test_tool_yaml_entry_no_mcp():
    """ToolYamlEntry.mcp should be None when not present in YAML."""
    catalog = ToolCatalog(raw_tools={
        "think": {"description": "Think"},
    })

    entry = catalog.get("think")
    assert entry.mcp is None


# ---------------------------------------------------------------------------
# AgentCatalog MCP server tests
# ---------------------------------------------------------------------------


def test_agent_catalog_get_mcp_server():
    """get_mcp_server() resolves tool groups and returns MCPServerConfig."""
    catalog = AgentCatalog(
        tool_groups={
            "graph_read": ["discover", "search", "read"],
        },
        raw_agents={},
        raw_mcp_servers={
            "knowledge_graph": {
                "description": "My knowledge graph",
                "tools": [
                    {"group": "graph_read"},
                    "get_tags",
                ],
                "write_tools": [
                    "create_page",
                ],
                "resources": [
                    {"uri": "test://project", "description": "Project"},
                ],
                "prompts": [
                    {"name": "research", "description": "Research", "arguments": ["topic"]},
                ],
            },
        },
    )

    config = catalog.get_mcp_server("knowledge_graph")
    assert config.name == "knowledge_graph"
    assert config.description == "My knowledge graph"
    assert config.tools == ["discover", "search", "read", "get_tags"]
    assert config.write_tools == ["create_page"]
    assert len(config.resources) == 1
    assert config.resources[0].uri == "test://project"
    assert len(config.prompts) == 1
    assert config.prompts[0].name == "research"


def test_agent_catalog_get_mcp_server_not_found():
    """get_mcp_server() raises KeyError for unknown server."""
    catalog = AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers={},
    )

    with pytest.raises(KeyError, match="not found"):
        catalog.get_mcp_server("nonexistent")


def test_agent_catalog_list_mcp_servers():
    """list_mcp_servers() returns all server names."""
    catalog = AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers={
            "server_a": {"description": "A"},
            "server_b": {"description": "B"},
        },
    )

    assert sorted(catalog.list_mcp_servers()) == ["server_a", "server_b"]


def test_agent_catalog_mcp_server_backward_compat():
    """AgentCatalog works fine without mcp_servers (backward compat)."""
    catalog = AgentCatalog(
        tool_groups={"test": ["a", "b"]},
        raw_agents={"my_agent": {"model": "fast", "description": "Test"}},
    )

    assert catalog.list_mcp_servers() == []
    assert catalog.list_agents() == ["my_agent"]
