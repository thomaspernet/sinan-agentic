"""Tool Registry - Centralized definition of all available tools."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List


@dataclass
class ToolDefinition:
    """Schema for a single tool."""
    name: str
    description: str
    function: Callable
    category: str  # "search", "analysis", "refinement"
    parameters_description: str
    returns_description: str
    recovery_hint: str = ""  # Guidance shown to agent when this tool errors


@dataclass
class ToolRegistry:
    """Central registry of all tools available to agents.
    
    Dataclass-driven design makes it easy to add new tools.
    """
    
    # Store tools by category
    _tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    
    def register(self, tool_def: ToolDefinition):
        """Register a new tool."""
        self._tools[tool_def.name] = tool_def
    
    def get_tool(self, name: str) -> ToolDefinition:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]
    
    def get_tool_functions(self, tool_names: List[str]) -> List[Callable]:
        """Get actual function objects for given tool names."""
        return [self._tools[name].function for name in tool_names if name in self._tools]
    
    def to_instruction_text(self, tool_names: List[str] = None) -> str:
        """Convert tools to instruction text for agent prompts.
        
        Args:
            tool_names: Specific tools to include, or None for all
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        
        text = "## Available Tools\n\n"
        
        # Group by category
        by_category: Dict[str, List[ToolDefinition]] = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        # Format by category
        for category, category_tools in sorted(by_category.items()):
            text += f"### {category.title()} Tools\n\n"
            for tool in category_tools:
                text += f"**{tool.name}**\n"
                text += f"- {tool.description}\n"
                text += f"- Parameters: {tool.parameters_description}\n"
                text += f"- Returns: {tool.returns_description}\n\n"
        
        return text


# Global tool registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tool(
    name: str,
    description: str,
    category: str,
    parameters_description: str,
    returns_description: str,
    recovery_hint: str = "",
):
    """Decorator to register a tool.

    Usage:
        @register_tool(
            name="execute_cypher",
            description="Execute Cypher query",
            category="search",
            parameters_description="query (str): Cypher query",
            returns_description="Dict with success and results",
            recovery_hint="Check that the query is valid Cypher syntax.",
        )
        @function_tool
        async def execute_cypher(ctx, query: str) -> dict:
            ...
    """
    def decorator(func):
        tool_def = ToolDefinition(
            name=name,
            description=description,
            function=func,
            category=category,
            parameters_description=parameters_description,
            returns_description=returns_description,
            recovery_hint=recovery_hint,
        )
        _global_registry.register(tool_def)
        return func
    return decorator
