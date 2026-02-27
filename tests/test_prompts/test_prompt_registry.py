"""Tests for PromptRegistry — prompt template management and versioning.

Covers:
- register() — basic, auto-variable detection, with metadata, multiple versions
- get() — latest, specific version, missing name, missing version
- list_versions() — ordering, missing name
- list_prompts() — empty, populated
- render() — happy path, missing variable, latest version
- diff() — identical, added vars, removed vars, missing name/version
- PromptTemplate.render() — direct use
- PromptVersion immutability
- PromptStorageBackend extension point
- Edge cases — empty template, whitespace, single character
"""
from __future__ import annotations

import pytest

from agent_observability.prompts import (
    PromptDiff,
    PromptNotFoundError,
    PromptRegistry,
    PromptRegistryError,
    PromptTemplate,
    PromptVersion,
    PromptVersionNotFoundError,
)
from agent_observability.prompts.prompt_registry import (
    PromptStorageBackend,
    _extract_template_variables,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> PromptRegistry:
    return PromptRegistry()


# ---------------------------------------------------------------------------
# _extract_template_variables
# ---------------------------------------------------------------------------


class TestExtractTemplateVariables:
    def test_single_variable(self) -> None:
        result = _extract_template_variables("Hello, {name}!")
        assert result == ["name"]

    def test_multiple_variables(self) -> None:
        result = _extract_template_variables("{greeting}, {name}! You are {age}.")
        assert result == ["greeting", "name", "age"]

    def test_no_variables(self) -> None:
        assert _extract_template_variables("Hello world") == []

    def test_duplicate_variable_deduplicated(self) -> None:
        result = _extract_template_variables("{x} and {x} again")
        assert result == ["x"]

    def test_empty_string(self) -> None:
        assert _extract_template_variables("") == []

    def test_preserves_first_occurrence_order(self) -> None:
        result = _extract_template_variables("{b} {a} {c} {b}")
        assert result == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# PromptRegistry.register()
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_prompt_version(self) -> None:
        registry = _make_registry()
        version = registry.register("greet", "Hello, {name}!")
        assert isinstance(version, PromptVersion)
        assert version.name == "greet"
        assert version.version == 1
        assert version.template == "Hello, {name}!"

    def test_first_version_is_one(self) -> None:
        registry = _make_registry()
        version = registry.register("test", "template")
        assert version.version == 1

    def test_second_registration_increments_version(self) -> None:
        registry = _make_registry()
        registry.register("test", "v1 template")
        v2 = registry.register("test", "v2 template")
        assert v2.version == 2

    def test_multiple_versions_incremented_correctly(self) -> None:
        registry = _make_registry()
        for i in range(5):
            v = registry.register("prompt", f"template {i}")
            assert v.version == i + 1

    def test_auto_detects_variables(self) -> None:
        registry = _make_registry()
        version = registry.register("greet", "Hello, {name}! You are {age}.")
        assert "name" in version.variables
        assert "age" in version.variables

    def test_explicit_variables_override_auto_detection(self) -> None:
        registry = _make_registry()
        version = registry.register(
            "test", "Hello, {name}!", variables=["name", "extra"]
        )
        assert version.variables == ["name", "extra"]

    def test_metadata_stored(self) -> None:
        registry = _make_registry()
        version = registry.register(
            "test", "template", metadata={"author": "Alice", "tag": "v1"}
        )
        assert version.metadata["author"] == "Alice"
        assert version.metadata["tag"] == "v1"

    def test_metadata_defaults_to_empty_dict(self) -> None:
        registry = _make_registry()
        version = registry.register("test", "template")
        assert version.metadata == {}

    def test_created_at_is_set(self) -> None:
        from datetime import timezone
        registry = _make_registry()
        version = registry.register("test", "template")
        assert version.created_at.tzinfo is not None
        assert version.created_at.tzinfo == timezone.utc

    def test_different_prompts_have_independent_versions(self) -> None:
        registry = _make_registry()
        registry.register("a", "template a1")
        registry.register("a", "template a2")
        b = registry.register("b", "template b1")
        assert b.version == 1

    def test_empty_template_allowed(self) -> None:
        registry = _make_registry()
        version = registry.register("empty", "")
        assert version.template == ""
        assert version.variables == []


# ---------------------------------------------------------------------------
# PromptRegistry.get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_latest_version(self) -> None:
        registry = _make_registry()
        registry.register("greet", "v1")
        registry.register("greet", "v2")
        template = registry.get("greet")
        assert template.version == 2
        assert template.template == "v2"

    def test_get_specific_version(self) -> None:
        registry = _make_registry()
        registry.register("greet", "v1")
        registry.register("greet", "v2")
        template = registry.get("greet", version=1)
        assert template.version == 1
        assert template.template == "v1"

    def test_get_returns_prompt_template(self) -> None:
        registry = _make_registry()
        registry.register("greet", "Hello, {name}!")
        template = registry.get("greet")
        assert isinstance(template, PromptTemplate)

    def test_get_missing_name_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(PromptNotFoundError) as exc_info:
            registry.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_missing_version_raises(self) -> None:
        registry = _make_registry()
        registry.register("greet", "v1")
        with pytest.raises(PromptVersionNotFoundError) as exc_info:
            registry.get("greet", version=99)
        assert "greet" in str(exc_info.value)
        assert "99" in str(exc_info.value)

    def test_get_version_none_returns_latest(self) -> None:
        registry = _make_registry()
        for i in range(3):
            registry.register("test", f"version {i+1}")
        template = registry.get("test", version=None)
        assert template.version == 3


# ---------------------------------------------------------------------------
# PromptRegistry.list_versions()
# ---------------------------------------------------------------------------


class TestListVersions:
    def test_single_version(self) -> None:
        registry = _make_registry()
        registry.register("prompt", "v1")
        versions = registry.list_versions("prompt")
        assert len(versions) == 1
        assert versions[0].version == 1

    def test_multiple_versions_ordered_oldest_first(self) -> None:
        registry = _make_registry()
        for i in range(4):
            registry.register("prompt", f"template {i+1}")
        versions = registry.list_versions("prompt")
        assert [v.version for v in versions] == [1, 2, 3, 4]

    def test_missing_name_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(PromptNotFoundError):
            registry.list_versions("nonexistent")

    def test_returns_copy_not_reference(self) -> None:
        registry = _make_registry()
        registry.register("p", "template")
        versions1 = registry.list_versions("p")
        versions1.clear()
        versions2 = registry.list_versions("p")
        assert len(versions2) == 1


# ---------------------------------------------------------------------------
# PromptRegistry.list_prompts()
# ---------------------------------------------------------------------------


class TestListPrompts:
    def test_empty_registry(self) -> None:
        registry = _make_registry()
        assert registry.list_prompts() == []

    def test_returns_sorted_names(self) -> None:
        registry = _make_registry()
        registry.register("bravo", "t")
        registry.register("alpha", "t")
        registry.register("charlie", "t")
        assert registry.list_prompts() == ["alpha", "bravo", "charlie"]

    def test_no_duplicates_for_multiple_versions(self) -> None:
        registry = _make_registry()
        registry.register("prompt", "v1")
        registry.register("prompt", "v2")
        assert registry.list_prompts() == ["prompt"]


# ---------------------------------------------------------------------------
# PromptRegistry.render()
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_latest(self) -> None:
        registry = _make_registry()
        registry.register("greet", "Hello, {name}!")
        result = registry.render("greet", name="Alice")
        assert result == "Hello, Alice!"

    def test_render_specific_version(self) -> None:
        registry = _make_registry()
        registry.register("greet", "Hello, {name}!")
        registry.register("greet", "Hi, {name}! How are you?")
        result = registry.render("greet", version=1, name="Bob")
        assert result == "Hello, Bob!"

    def test_render_multiple_variables(self) -> None:
        registry = _make_registry()
        registry.register("intro", "I am {name}, aged {age}, from {city}.")
        result = registry.render("intro", name="Alice", age=30, city="London")
        assert result == "I am Alice, aged 30, from London."

    def test_render_missing_variable_raises_key_error(self) -> None:
        registry = _make_registry()
        registry.register("greet", "Hello, {name}!")
        with pytest.raises(KeyError):
            registry.render("greet")

    def test_render_missing_prompt_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(PromptNotFoundError):
            registry.render("nonexistent", name="Alice")

    def test_render_no_variables(self) -> None:
        registry = _make_registry()
        registry.register("static", "This is a static prompt.")
        result = registry.render("static")
        assert result == "This is a static prompt."


# ---------------------------------------------------------------------------
# PromptTemplate.render()
# ---------------------------------------------------------------------------


class TestPromptTemplateRender:
    def test_render_substitutes_variables(self) -> None:
        template = PromptTemplate(
            name="t", version=1, template="Hello, {name}!", variables=["name"]
        )
        assert template.render(name="World") == "Hello, World!"

    def test_render_missing_variable_raises(self) -> None:
        template = PromptTemplate(
            name="t", version=1, template="{a} and {b}", variables=["a", "b"]
        )
        with pytest.raises(KeyError):
            template.render(a="only_a")

    def test_render_extra_kwargs_ignored(self) -> None:
        template = PromptTemplate(
            name="t", version=1, template="Hello!", variables=[]
        )
        # str.format ignores extra kwargs
        result = template.render(unused="value")
        assert result == "Hello!"


# ---------------------------------------------------------------------------
# PromptRegistry.diff()
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_identical_templates(self) -> None:
        registry = _make_registry()
        registry.register("p", "Hello, {name}!")
        registry.register("p", "Hello, {name}!")
        diff = registry.diff("p", 1, 2)
        assert diff.is_identical
        assert diff.added_variables == []
        assert diff.removed_variables == []

    def test_diff_added_variable(self) -> None:
        registry = _make_registry()
        registry.register("p", "Hello, {name}!")
        registry.register("p", "Hello, {name}! You are {age}.")
        diff = registry.diff("p", 1, 2)
        assert not diff.is_identical
        assert "age" in diff.added_variables
        assert diff.removed_variables == []

    def test_diff_removed_variable(self) -> None:
        registry = _make_registry()
        registry.register("p", "Hello, {name}! You are {age}.")
        registry.register("p", "Hello, {name}!")
        diff = registry.diff("p", 1, 2)
        assert not diff.is_identical
        assert "age" in diff.removed_variables
        assert diff.added_variables == []

    def test_diff_contains_diff_lines(self) -> None:
        registry = _make_registry()
        registry.register("p", "Line one\nLine two\n")
        registry.register("p", "Line one\nLine THREE\n")
        diff = registry.diff("p", 1, 2)
        assert len(diff.diff_lines) > 0
        # At least one line should show a removal
        assert any(line.startswith("-") for line in diff.diff_lines)

    def test_diff_records_correct_versions(self) -> None:
        registry = _make_registry()
        registry.register("p", "v1")
        registry.register("p", "v2")
        diff = registry.diff("p", 1, 2)
        assert diff.version_a == 1
        assert diff.version_b == 2
        assert diff.name == "p"

    def test_diff_missing_prompt_raises(self) -> None:
        registry = _make_registry()
        with pytest.raises(PromptNotFoundError):
            registry.diff("nonexistent", 1, 2)

    def test_diff_missing_version_a_raises(self) -> None:
        registry = _make_registry()
        registry.register("p", "template")
        with pytest.raises(PromptVersionNotFoundError):
            registry.diff("p", 99, 1)

    def test_diff_missing_version_b_raises(self) -> None:
        registry = _make_registry()
        registry.register("p", "template")
        with pytest.raises(PromptVersionNotFoundError):
            registry.diff("p", 1, 99)

    def test_diff_templates_stored_correctly(self) -> None:
        registry = _make_registry()
        registry.register("p", "template A")
        registry.register("p", "template B")
        diff = registry.diff("p", 1, 2)
        assert diff.template_a == "template A"
        assert diff.template_b == "template B"


# ---------------------------------------------------------------------------
# PromptStorageBackend extension point
# ---------------------------------------------------------------------------


class TestPromptStorageBackend:
    def test_save_and_get_all(self) -> None:
        backend = PromptStorageBackend()
        from datetime import datetime, timezone
        pv = PromptVersion(
            name="test", version=1, template="hello", variables=[],
            metadata={}, created_at=datetime.now(tz=timezone.utc)
        )
        backend.save(pv)
        result = backend.get_all("test")
        assert len(result) == 1
        assert result[0].name == "test"

    def test_get_all_unknown_returns_empty(self) -> None:
        backend = PromptStorageBackend()
        assert backend.get_all("unknown") == []

    def test_list_names_sorted(self) -> None:
        backend = PromptStorageBackend()
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        for name in ["z", "a", "m"]:
            backend.save(PromptVersion(
                name=name, version=1, template="t", variables=[],
                metadata={}, created_at=now
            ))
        assert backend.list_names() == ["a", "m", "z"]

    def test_custom_backend_used_by_registry(self) -> None:
        """Registry delegates to the provided backend."""
        class CountingBackend(PromptStorageBackend):
            save_count = 0
            def save(self, pv: PromptVersion) -> None:
                CountingBackend.save_count += 1
                super().save(pv)

        backend = CountingBackend()
        registry = PromptRegistry(backend=backend)
        registry.register("p", "t")
        assert CountingBackend.save_count == 1


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_prompt_not_found_is_registry_error(self) -> None:
        exc = PromptNotFoundError("test")
        assert isinstance(exc, PromptRegistryError)

    def test_version_not_found_is_registry_error(self) -> None:
        exc = PromptVersionNotFoundError("test", 1)
        assert isinstance(exc, PromptRegistryError)

    def test_prompt_not_found_message(self) -> None:
        exc = PromptNotFoundError("my_prompt")
        assert "my_prompt" in str(exc)

    def test_version_not_found_message(self) -> None:
        exc = PromptVersionNotFoundError("my_prompt", 5)
        assert "my_prompt" in str(exc)
        assert "5" in str(exc)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_large_number_of_versions(self) -> None:
        registry = _make_registry()
        for i in range(20):
            registry.register("bulk", f"template {i}")
        versions = registry.list_versions("bulk")
        assert len(versions) == 20
        assert versions[-1].version == 20

    def test_template_with_only_whitespace(self) -> None:
        registry = _make_registry()
        version = registry.register("ws", "   ")
        assert version.template == "   "

    def test_metadata_is_copied_not_shared(self) -> None:
        """Mutating the original metadata dict should not affect stored version."""
        registry = _make_registry()
        meta = {"key": "value"}
        version = registry.register("p", "t", metadata=meta)
        meta["key"] = "modified"
        stored = registry.list_versions("p")[0]
        assert stored.metadata["key"] == "value"
