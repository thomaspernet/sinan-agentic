"""Tests for MCP YAML schema models."""

from agents_core.mcp.yaml_schema import (
    MCPAnnotationsConfig,
    MCPPromptConfig,
    MCPResourceConfig,
    MCPServerConfig,
    MCPToolConfig,
)


def test_mcp_tool_config_defaults():
    cfg = MCPToolConfig()
    assert cfg.expose is False
    assert cfg.annotations.readOnlyHint is None


def test_mcp_tool_config_expose():
    cfg = MCPToolConfig(
        expose=True,
        annotations=MCPAnnotationsConfig(readOnlyHint=True, openWorldHint=False),
    )
    assert cfg.expose is True
    assert cfg.annotations.readOnlyHint is True
    assert cfg.annotations.openWorldHint is False
    assert cfg.annotations.destructiveHint is None


def test_mcp_server_config():
    cfg = MCPServerConfig(
        name="test_server",
        description="A test server",
        tools=["discover", "search"],
        write_tools=["create_page"],
        resources=[MCPResourceConfig(uri="test://doc/{uuid}", description="A doc")],
        prompts=[MCPPromptConfig(name="research", arguments=["topic"])],
    )
    assert cfg.name == "test_server"
    assert len(cfg.tools) == 2
    assert len(cfg.write_tools) == 1
    assert cfg.resources[0].uri == "test://doc/{uuid}"
    assert cfg.prompts[0].name == "research"
    assert cfg.prompts[0].arguments == ["topic"]


def test_mcp_server_config_defaults():
    cfg = MCPServerConfig(name="minimal")
    assert cfg.tools == []
    assert cfg.write_tools == []
    assert cfg.resources == []
    assert cfg.prompts == []
