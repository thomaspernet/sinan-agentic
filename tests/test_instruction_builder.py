"""Tests for InstructionBuilder base class."""

from agents_core.instructions import InstructionBuilder


class TestBuild:
    """Test the build/assembly mechanism."""

    def test_empty_builder_produces_empty_string(self):
        builder = InstructionBuilder(None, None)
        assert builder.build() == ""

    def test_single_section_override(self):
        class PersonaOnly(InstructionBuilder):
            def persona(self):
                return "You are a test agent."

        result = PersonaOnly(None, None).build()
        assert result == "You are a test agent."

    def test_none_sections_skipped(self):
        class TwoSections(InstructionBuilder):
            def persona(self):
                return "You are a test agent."

            def rules(self):
                return "Rules:\n- Be good."

        result = TwoSections(None, None).build()
        assert "You are a test agent." in result
        assert "Rules:\n- Be good." in result
        # context_section, steps, output_format are None -> skipped
        assert result.count("\n\n") == 1  # exactly one separator

    def test_all_sections_assembled(self):
        class AllSections(InstructionBuilder):
            def persona(self):
                return "Persona."

            def context_section(self):
                return "Context."

            def steps(self):
                return "Steps."

            def rules(self):
                return "Rules."

            def output_format(self):
                return "Output."

        result = AllSections(None, None).build()
        assert result == "Persona.\n\nContext.\n\nSteps.\n\nRules.\n\nOutput."

    def test_extra_sections_appended(self):
        class WithExtras(InstructionBuilder):
            def persona(self):
                return "Persona."

            def extra_sections(self):
                return [("Principles", "Be helpful.")]

        result = WithExtras(None, None).build()
        assert "Persona." in result
        assert "Principles\nBe helpful." in result

    def test_custom_sections_order(self):
        class Reordered(InstructionBuilder):
            def persona(self):
                return "P"

            def rules(self):
                return "R"

            def sections(self):
                return ["rules", "persona"]

        result = Reordered(None, None).build()
        assert result == "R\n\nP"

    def test_whitespace_stripped(self):
        class Padded(InstructionBuilder):
            def persona(self):
                return "  Persona.  \n"

        result = Padded(None, None).build()
        assert result == "Persona."


class TestCtxAttr:
    """Test context attribute access."""

    def test_none_context_returns_default(self):
        builder = InstructionBuilder(None, None)
        assert builder._ctx_attr("missing", "fallback") == "fallback"

    def test_direct_context(self):
        class FakeCtx:
            name = "test"

        builder = InstructionBuilder(FakeCtx(), None)
        assert builder._ctx_attr("name") == "test"

    def test_unwraps_wrapper(self):
        class Inner:
            name = "inner_value"

        class Wrapper:
            context = Inner()

        wrapper = Wrapper()
        builder = InstructionBuilder(wrapper, None)
        assert builder.ctx is wrapper.context
        assert builder._ctx_attr("name") == "inner_value"

    def test_missing_attr_returns_default(self):
        class FakeCtx:
            pass

        builder = InstructionBuilder(FakeCtx(), None)
        assert builder._ctx_attr("missing", 42) == 42


class TestFormatSteps:
    def test_basic(self):
        result = InstructionBuilder.format_steps(["Do X.", "Do Y."])
        assert result == "Steps:\n1. Do X.\n2. Do Y."

    def test_custom_start(self):
        result = InstructionBuilder.format_steps(["First."], start=0)
        assert result == "Steps:\n0. First."


class TestFormatRules:
    def test_basic(self):
        result = InstructionBuilder.format_rules(["No lying.", "Be concise."])
        assert result == "Rules:\n- No lying.\n- Be concise."


class TestFormatPersona:
    def test_override_returned_directly(self):
        result = InstructionBuilder.format_persona("ignored", "Custom persona.")
        assert result == "Custom persona."

    def test_role_with_consonant(self):
        result = InstructionBuilder.format_persona("test specialist")
        assert result == "You are a test specialist."

    def test_role_with_vowel(self):
        result = InstructionBuilder.format_persona("expert reviewer")
        assert result == "You are an expert reviewer."


class TestCallable:
    def test_returns_function(self):
        fn = InstructionBuilder.callable()
        assert callable(fn)

    def test_function_produces_output(self):
        class Simple(InstructionBuilder):
            def persona(self):
                return "Hello."

        fn = Simple.callable()
        result = fn(None, None)
        assert result == "Hello."

    def test_function_signature(self):
        """The callable accepts (context, agent_def) matching base_runner."""
        fn = InstructionBuilder.callable()
        # Both args optional
        result = fn()
        assert result == ""
        result = fn(None)
        assert result == ""
        result = fn(None, None)
        assert result == ""


class TestTopLevelImport:
    def test_importable_from_agents_core(self):
        from agents_core import InstructionBuilder as IB
        assert IB is InstructionBuilder

    def test_importable_from_instructions(self):
        from agents_core.instructions import InstructionBuilder as IB
        assert IB is InstructionBuilder
