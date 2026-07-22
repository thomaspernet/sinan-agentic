"""Tests for MCP YAML schema models."""

import pytest
from pydantic import ValidationError

from sinan_agentic_core.mcp.yaml_schema import (
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


class TestUnrecognizedKeys:
    """Every model gates its own block — a typo fails instead of being dropped."""

    def test_tool_config_rejects_an_unknown_key(self):
        with pytest.raises(ValidationError, match="exposed"):
            MCPToolConfig(exposed=True)

    def test_annotations_reject_an_unknown_hint(self):
        """A hint this model does not carry cannot reach a client — say so."""
        with pytest.raises(ValidationError, match="titleHint"):
            MCPAnnotationsConfig(titleHint="Search")

    def test_resource_config_rejects_an_unknown_key(self):
        with pytest.raises(ValidationError, match="descriptions"):
            MCPResourceConfig(uri="test://doc", descriptions="A doc")

    def test_prompt_config_rejects_an_unknown_key(self):
        with pytest.raises(ValidationError, match="args"):
            MCPPromptConfig(name="research", args=["topic"])

    def test_server_config_rejects_an_unknown_key(self):
        with pytest.raises(ValidationError, match="write-tools"):
            MCPServerConfig(name="minimal", **{"write-tools": ["create_page"]})
