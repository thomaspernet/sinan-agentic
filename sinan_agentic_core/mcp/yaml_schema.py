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

from pydantic import BaseModel, ConfigDict


class MCPAnnotationsConfig(BaseModel):
    """MCP tool annotations from tools.yaml.

    Field names mirror the MCP protocol's camelCase wire format, so ruff's
    snake_case rule (N815) is suppressed per-line.
    """

    # Mirroring a wire format does not make this a decoder: it validates what an
    # author wrote in tools.yaml, never a payload received from a peer, so
    # forward compatibility is not its problem. The server builder forwards
    # exactly these four hints (``server_builder.py``), so a newer protocol
    # annotation accepted here would be dropped before reaching a client —
    # rejecting it points the author at the real fix, adding the field.
    model_config = ConfigDict(extra="forbid")

    readOnlyHint: bool | None = None  # noqa: N815 — MCP protocol field name
    openWorldHint: bool | None = None  # noqa: N815 — MCP protocol field name
    destructiveHint: bool | None = None  # noqa: N815 — MCP protocol field name
    idempotentHint: bool | None = None  # noqa: N815 — MCP protocol field name


class MCPToolConfig(BaseModel):
    """Per-tool MCP config from the ``mcp`` section in tools.yaml."""

    # An unknown key is a typo, and an accepted ``exposed: true`` would leave the
    # tool unexposed while reading as opted in — so reject it.
    model_config = ConfigDict(extra="forbid")

    expose: bool = False
    annotations: MCPAnnotationsConfig = MCPAnnotationsConfig()


class MCPResourceConfig(BaseModel):
    """MCP resource definition from agents.yaml ``mcp_servers`` section."""

    # ``AgentCatalog.get_mcp_server()`` builds this straight from each entry in
    # the ``resources:`` list, so an unknown key is a typo — a mistyped
    # ``description`` would register the resource with an empty one.
    model_config = ConfigDict(extra="forbid")

    uri: str
    description: str = ""


class MCPPromptConfig(BaseModel):
    """MCP prompt definition from agents.yaml ``mcp_servers`` section."""

    # Built straight from each entry in the ``prompts:`` list, same as
    # :class:`MCPResourceConfig` — an unknown key is a typo, so reject it.
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    arguments: list[str] = []


class MCPServerConfig(BaseModel):
    """Resolved MCP server definition.

    Tools are plain string lists (groups already expanded, conditions evaluated).
    """

    # ``AgentCatalog.get_mcp_server()`` forwards the whole raw server block here
    # so this model is the gate it passes through — an unknown key is a typo
    # (``write-tools`` for ``write_tools`` silently exposes nothing), and
    # ``name`` is supplied from the mapping key rather than declared in the block.
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    tools: list[str] = []
    write_tools: list[str] = []
    resources: list[MCPResourceConfig] = []
    prompts: list[MCPPromptConfig] = []
