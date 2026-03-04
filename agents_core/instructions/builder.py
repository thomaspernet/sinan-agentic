"""Base class for agent instruction builders.

All agent instructions follow a consistent section pattern:
persona, context, steps, rules, output format. Subclasses override
the section methods they need; unused sections return None and are
skipped during assembly.

Usage:
    class MyBuilder(InstructionBuilder):
        def persona(self) -> str:
            return "You are a ..."

        def steps(self) -> str:
            return self.format_steps(["Do X.", "Do Y."])

    # In agent definition:
    my_agent = AgentDefinition(
        instructions=MyBuilder.callable(),
        ...
    )

Why use InstructionBuilder:
    - Consistent section ordering across all agents (persona -> context -> steps -> rules -> output)
    - Sections returning None are silently skipped — no empty headers
    - Shared formatting utilities (format_steps, format_rules, format_persona)
    - Safe context access via _ctx_attr() with automatic RunContextWrapper unwrapping
    - callable() classmethod bridges to AgentDefinition.instructions
    - Subclasses override only what they need — minimal boilerplate
"""

from typing import Any


class InstructionBuilder:
    """Base instruction builder with section-based assembly.

    Subclasses override section methods to return content strings.
    Sections returning None are skipped. The build() method assembles
    all sections with consistent double-newline spacing.
    """

    def __init__(self, context: Any, agent_def: Any) -> None:
        self.ctx = getattr(context, "context", context) if context else None
        self.agent_def = agent_def

    # ------------------------------------------------------------------ #
    # Section methods — override in subclasses
    # ------------------------------------------------------------------ #

    def persona(self) -> str | None:
        """Opening identity statement. E.g., 'You are a ...'."""
        return None

    def context_section(self) -> str | None:
        """Environment info: filters, configs, strategy, hints."""
        return None

    def steps(self) -> str | None:
        """Numbered procedure the agent should follow."""
        return None

    def rules(self) -> str | None:
        """Bullet-point constraints."""
        return None

    def output_format(self) -> str | None:
        """Expected output structure (JSON template, etc.)."""
        return None

    def extra_sections(self) -> list[tuple[str, str]]:
        """Additional named sections appended after the main sections.

        Returns list of (header, body) tuples. Useful for agent-specific
        sections like principles or configuration blocks.
        """
        return []

    # ------------------------------------------------------------------ #
    # Section ordering
    # ------------------------------------------------------------------ #

    def sections(self) -> list[str]:
        """Ordered list of section method names to assemble.

        Override to reorder, add, or remove sections.
        Extra sections from extra_sections() are always appended after these.
        """
        return ["persona", "context_section", "steps", "rules", "output_format"]

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #

    def build(self) -> str:
        """Assemble all sections into the final instruction string."""
        parts: list[str] = []

        for section_name in self.sections():
            content = getattr(self, section_name)()
            if content is not None:
                parts.append(content.strip())

        for header, body in self.extra_sections():
            parts.append(f"{header}\n{body}".strip())

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Context access utilities
    # ------------------------------------------------------------------ #

    def _ctx_attr(self, name: str, default: Any = None) -> Any:
        """Safely read an attribute from the unwrapped context.

        Handles None context and missing attributes gracefully.
        The constructor automatically unwraps RunContextWrapper
        via getattr(context, "context", context).
        """
        if self.ctx is None:
            return default
        return getattr(self.ctx, name, default)

    # ------------------------------------------------------------------ #
    # Formatting utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_steps(step_list: list[str], start: int = 1) -> str:
        """Format step descriptions as a numbered list with header.

        Returns:
            "Steps:\\n1. First step\\n2. Second step"
        """
        lines = [f"{i}. {step}" for i, step in enumerate(step_list, start=start)]
        return "Steps:\n" + "\n".join(lines)

    @staticmethod
    def format_rules(rule_list: list[str]) -> str:
        """Format rule strings as a bulleted list with header.

        Returns:
            "Rules:\\n- First rule\\n- Second rule"
        """
        lines = [f"- {rule}" for rule in rule_list]
        return "Rules:\n" + "\n".join(lines)

    @staticmethod
    def format_persona(role: str, persona_override: str | None = None) -> str:
        """Build persona opening line from role or full override.

        If persona_override is provided, returns it directly.
        Otherwise builds "You are a/an {role}." with correct article.
        """
        if persona_override:
            return persona_override
        article = "an" if role and role[0].lower() in "aeiou" else "a"
        return f"You are {article} {role}."

    # ------------------------------------------------------------------ #
    # Agent definition bridge
    # ------------------------------------------------------------------ #

    @classmethod
    def callable(cls) -> Any:
        """Return a function compatible with AgentDefinition.instructions.

        The returned function has signature (context, agent_def) -> str,
        matching what BaseAgentRunner._build_instructions() expects.

        Usage:
            class MyBuilder(InstructionBuilder):
                def persona(self):
                    return "You are a ..."

            my_agent = AgentDefinition(
                instructions=MyBuilder.callable(),
                ...
            )
        """

        def _build_instructions(context: Any = None, agent_def: Any = None) -> str:
            return cls(context, agent_def).build()

        return _build_instructions
