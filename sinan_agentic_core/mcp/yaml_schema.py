"""Pydantic models for MCP YAML configuration.

Extends ``tools.yaml`` and ``agents.yaml`` with MCP-specific sections:

**tools.yaml** — per-tool MCP metadata::

    tools:
      discover:
        description: ...
        category: graph_navigation
        mcp:
          expose: true
          annotations:
            readOnlyHint: true

**agents.yaml** — MCP server definitions::

    mcp_servers:
      knowledge_graph:
        description: "My knowledge graph"
        tools:
          - group: graph_read
          - get_tags
        write_tools:
          - create_page
"""

from pydantic import BaseModel


class MCPAnnotationsConfig(BaseModel):
    """MCP tool annotations from tools.yaml."""

    readOnlyHint: bool | None = None
    openWorldHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None


class MCPToolConfig(BaseModel):
    """Per-tool MCP config from the ``mcp`` section in tools.yaml."""

    expose: bool = False
    annotations: MCPAnnotationsConfig = MCPAnnotationsConfig()


class MCPResourceConfig(BaseModel):
    """MCP resource definition from agents.yaml ``mcp_servers`` section."""

    uri: str
    description: str = ""


class MCPPromptConfig(BaseModel):
    """MCP prompt definition from agents.yaml ``mcp_servers`` section."""

    name: str
    description: str = ""
    arguments: list[str] = []


class MCPServerConfig(BaseModel):
    """Resolved MCP server definition.

    Tools are plain string lists (groups already expanded, conditions evaluated).
    """

    name: str
    description: str = ""
    tools: list[str] = []
    write_tools: list[str] = []
    resources: list[MCPResourceConfig] = []
    prompts: list[MCPPromptConfig] = []
