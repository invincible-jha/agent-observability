"""Comprehensive tests for the plugin registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


class BasePlugin(ABC):
    @abstractmethod
    def run(self) -> str: ...


class AnotherBase(ABC):
    @abstractmethod
    def execute(self) -> None: ...


@pytest.fixture()
def registry() -> PluginRegistry[BasePlugin]:
    return PluginRegistry(BasePlugin, "test-registry")


# ── PluginNotFoundError ────────────────────────────────────────────────────────


class TestPluginNotFoundError:
    def test_attributes(self) -> None:
        err = PluginNotFoundError("missing-plugin", "my-registry")
        assert err.plugin_name == "missing-plugin"
        assert err.registry_name == "my-registry"
        assert "missing-plugin" in str(err)


# ── PluginAlreadyRegisteredError ───────────────────────────────────────────────


class TestPluginAlreadyRegisteredError:
    def test_attributes(self) -> None:
        err = PluginAlreadyRegisteredError("dup-plugin", "my-registry")
        assert err.plugin_name == "dup-plugin"
        assert err.registry_name == "my-registry"
        assert "dup-plugin" in str(err)


# ── PluginRegistry — Registration ─────────────────────────────────────────────


class TestPluginRegistryRegistration:
    def test_register_decorator(self, registry: PluginRegistry[BasePlugin]) -> None:
        @registry.register("my-plugin")
        class MyPlugin(BasePlugin):
            def run(self) -> str:
                return "ok"

        assert "my-plugin" in registry
        assert registry.get("my-plugin") is MyPlugin

    def test_register_duplicate_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        @registry.register("dup")
        class PluginA(BasePlugin):
            def run(self) -> str:
                return "a"

        with pytest.raises(PluginAlreadyRegisteredError):
            @registry.register("dup")
            class PluginB(BasePlugin):
                def run(self) -> str:
                    return "b"

    def test_register_non_subclass_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        class NotAPlugin:
            pass

        with pytest.raises(TypeError):
            registry.register("bad")(NotAPlugin)  # type: ignore[arg-type]

    def test_register_class_direct(self, registry: PluginRegistry[BasePlugin]) -> None:
        class DirectPlugin(BasePlugin):
            def run(self) -> str:
                return "direct"

        registry.register_class("direct-plugin", DirectPlugin)
        assert registry.get("direct-plugin") is DirectPlugin

    def test_register_class_duplicate_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        class P1(BasePlugin):
            def run(self) -> str:
                return "p1"

        class P2(BasePlugin):
            def run(self) -> str:
                return "p2"

        registry.register_class("same-name", P1)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("same-name", P2)

    def test_register_class_non_subclass_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        class NotRelated:
            pass

        with pytest.raises(TypeError):
            registry.register_class("bad", NotRelated)  # type: ignore[arg-type]


# ── PluginRegistry — Deregistration ───────────────────────────────────────────


class TestPluginRegistryDeregistration:
    def test_deregister_removes_plugin(self, registry: PluginRegistry[BasePlugin]) -> None:
        @registry.register("to-remove")
        class P(BasePlugin):
            def run(self) -> str:
                return "x"

        registry.deregister("to-remove")
        assert "to-remove" not in registry

    def test_deregister_unknown_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        with pytest.raises(PluginNotFoundError):
            registry.deregister("does-not-exist")


# ── PluginRegistry — Lookup ───────────────────────────────────────────────────


class TestPluginRegistryLookup:
    def test_get_registered_plugin(self, registry: PluginRegistry[BasePlugin]) -> None:
        @registry.register("lookup-me")
        class P(BasePlugin):
            def run(self) -> str:
                return "found"

        cls = registry.get("lookup-me")
        instance = cls()
        assert instance.run() == "found"

    def test_get_unregistered_raises(self, registry: PluginRegistry[BasePlugin]) -> None:
        with pytest.raises(PluginNotFoundError):
            registry.get("nonexistent")

    def test_list_plugins_sorted(self, registry: PluginRegistry[BasePlugin]) -> None:
        for name in ("beta", "alpha", "gamma"):
            cls_name = name.capitalize()

            class DynPlugin(BasePlugin):
                def run(self) -> str:
                    return cls_name

            DynPlugin.__name__ = cls_name
            DynPlugin.__qualname__ = cls_name
            registry.register_class(name, DynPlugin)

        names = registry.list_plugins()
        assert names == sorted(names)
        assert "alpha" in names

    def test_contains(self, registry: PluginRegistry[BasePlugin]) -> None:
        @registry.register("check-me")
        class P(BasePlugin):
            def run(self) -> str:
                return "x"

        assert "check-me" in registry
        assert "not-here" not in registry

    def test_len(self, registry: PluginRegistry[BasePlugin]) -> None:
        assert len(registry) == 0

        @registry.register("p1")
        class P1(BasePlugin):
            def run(self) -> str:
                return "p1"

        @registry.register("p2")
        class P2(BasePlugin):
            def run(self) -> str:
                return "p2"

        assert len(registry) == 2

    def test_repr(self, registry: PluginRegistry[BasePlugin]) -> None:
        r = repr(registry)
        assert "PluginRegistry" in r
        assert "test-registry" in r


# ── PluginRegistry — Entry-points ─────────────────────────────────────────────


class TestPluginRegistryEntrypoints:
    def test_load_entrypoints_skips_already_registered(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        class P(BasePlugin):
            def run(self) -> str:
                return "ep"

        registry.register_class("ep-plugin", P)

        mock_ep = MagicMock()
        mock_ep.name = "ep-plugin"
        mock_ep.load.return_value = P

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_observability.plugins")
        # Still only one registration
        assert len(registry) == 1

    def test_load_entrypoints_loads_new_plugin(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        class NewPlugin(BasePlugin):
            def run(self) -> str:
                return "new"

        mock_ep = MagicMock()
        mock_ep.name = "new-ep-plugin"
        mock_ep.load.return_value = NewPlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_observability.plugins")

        assert "new-ep-plugin" in registry

    def test_load_entrypoints_handles_load_failure(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "broken-ep"
        mock_ep.load.side_effect = ImportError("missing dependency")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            # Should not raise
            registry.load_entrypoints("agent_observability.plugins")

        assert "broken-ep" not in registry

    def test_load_entrypoints_handles_type_error(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        class BadPlugin:  # Not a subclass of BasePlugin
            pass

        mock_ep = MagicMock()
        mock_ep.name = "bad-ep"
        mock_ep.load.return_value = BadPlugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            # Should not raise, just skip
            registry.load_entrypoints("agent_observability.plugins")

        assert "bad-ep" not in registry
