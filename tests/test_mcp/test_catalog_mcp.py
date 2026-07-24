"""Tests for MCP extensions to ToolCatalog and AgentCatalog."""

import pytest
from pydantic import ValidationError

from sinan_agentic_core.registry.agent_catalog import AgentCatalog
from sinan_agentic_core.registry.tool_catalog import ToolCatalog

# ---------------------------------------------------------------------------
# ToolCatalog MCP tests
# ---------------------------------------------------------------------------


def test_tool_catalog_get_mcp_tools():
    """get_mcp_tools() returns only tools with mcp.expose: true."""
    catalog = ToolCatalog(
        raw_tools={
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
        }
    )

    mcp_tools = catalog.get_mcp_tools()
    assert sorted(mcp_tools) == ["discover", "search"]


def test_tool_catalog_get_mcp_tools_empty():
    """get_mcp_tools() returns empty list when no tools have mcp section."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think"},
        }
    )
    assert catalog.get_mcp_tools() == []


def test_tool_catalog_mcp_empty_block_is_a_declaration():
    """``mcp: {}`` resolves to ToolMCPConfig defaults, not to an absent block."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think", "mcp": {}},
        }
    )

    entry = catalog.get("think")
    assert entry.mcp is not None
    assert entry.mcp.expose is False
    assert entry.mcp.annotations == {}
    # The block declares MCP config with defaults, and the default is not exposed.
    assert catalog.get_mcp_tools() == []


def test_tool_catalog_mcp_null_block_opts_out():
    """An explicitly null ``mcp`` key means the tool declares no MCP config."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think", "mcp": None},
        }
    )

    assert catalog.get("think").mcp is None
    assert catalog.get_mcp_tools() == []


def test_tool_catalog_get_mcp_tools_validates_expose():
    """``expose`` is validated as a bool, not read for truthiness."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think", "mcp": {"expose": "no"}},
        }
    )

    assert catalog.get_mcp_tools() == []


def test_tool_catalog_get_mcp_tools_rejects_malformed_expose():
    """A non-boolean ``expose`` surfaces as a validation error, not as exposure."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think", "mcp": {"expose": "maybe"}},
        }
    )

    with pytest.raises(ValidationError):
        catalog.get_mcp_tools()


def test_tool_yaml_entry_mcp_field():
    """ToolYamlEntry should include mcp config when present."""
    catalog = ToolCatalog(
        raw_tools={
            "discover": {
                "description": "Discover",
                "mcp": {
                    "expose": True,
                    "annotations": {"readOnlyHint": True},
                },
            },
        }
    )

    entry = catalog.get("discover")
    assert entry.mcp is not None
    assert entry.mcp.expose is True
    assert entry.mcp.annotations == {"readOnlyHint": True}


def test_tool_yaml_entry_no_mcp():
    """ToolYamlEntry.mcp should be None when not present in YAML."""
    catalog = ToolCatalog(
        raw_tools={
            "think": {"description": "Think"},
        }
    )

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


def test_agent_catalog_get_mcp_server_rejects_an_unrecognized_key():
    """The whole raw block reaches the model, so a typo fails instead of exposing nothing."""
    catalog = AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers={
            "knowledge_graph": {
                "description": "My knowledge graph",
                "write-tools": ["create_page"],
            },
        },
    )

    with pytest.raises(ValidationError, match="write-tools"):
        catalog.get_mcp_server("knowledge_graph")


def test_agent_catalog_get_mcp_server_rejects_an_unrecognized_resource_key():
    """Nested entries are validated by the model's own fields, on the same terms."""
    catalog = AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers={
            "knowledge_graph": {
                "resources": [{"uri": "test://project", "descriptions": "Project"}],
            },
        },
    )

    with pytest.raises(ValidationError, match="descriptions"):
        catalog.get_mcp_server("knowledge_graph")


def test_agent_catalog_get_mcp_server_ignores_a_late_edit_to_the_callers_block():
    """The catalog copies the server blocks, so a caller's later edit does not reach them."""
    raw_mcp_servers = {
        "knowledge_graph": {
            "description": "My knowledge graph",
            "tools": ["discover"],
        },
    }
    catalog = AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers=raw_mcp_servers,
    )

    raw_mcp_servers["knowledge_graph"]["tools"].append("search")
    raw_mcp_servers["knowledge_graph"]["description"] = "Edited late"

    config = catalog.get_mcp_server("knowledge_graph")
    assert config.tools == ["discover"]
    assert config.description == "My knowledge graph"


def _catalog_with_nested_server_block() -> AgentCatalog:
    """A catalog whose one server block nests a mutable value under every model field."""
    return AgentCatalog(
        tool_groups={},
        raw_agents={},
        raw_mcp_servers={
            "knowledge_graph": {
                "description": "My knowledge graph",
                "tools": ["discover"],
                "resources": [{"uri": "test://project", "description": "Project"}],
                "prompts": [{"name": "research", "arguments": ["topic"]}],
            },
        },
    )


def test_agent_catalog_get_mcp_server_ignores_an_edit_inside_a_resolved_config():
    """The block is copied on the way out, so editing a resolved config cannot reach the catalog."""
    catalog = _catalog_with_nested_server_block()

    catalog.get_mcp_server("knowledge_graph").prompts[0].arguments.append("late")

    assert catalog.get_mcp_server("knowledge_graph").prompts[0].arguments == ["topic"]


def test_agent_catalog_get_mcp_server_two_configs_do_not_share_a_nested_value():
    """Each call resolves off its own copy, so one config's edit is invisible to the other."""
    catalog = _catalog_with_nested_server_block()
    first = catalog.get_mcp_server("knowledge_graph")
    second = catalog.get_mcp_server("knowledge_graph")

    first.prompts[0].arguments.append("late")

    assert second.prompts[0].arguments == ["topic"]


def test_agent_catalog_get_mcp_server_declared_values_survive_the_copy():
    """The copy detaches the block without dropping anything declared in it."""
    config = _catalog_with_nested_server_block().get_mcp_server("knowledge_graph")

    assert config.description == "My knowledge graph"
    assert config.tools == ["discover"]
    assert config.resources[0].uri == "test://project"
    assert config.prompts[0].arguments == ["topic"]


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
