"""PromptRegistry — prompt template management and versioning.

Provides version-tracked storage of prompt templates with rendering
and diffing capabilities.  All state is held in memory by default;
the ``_store`` attribute is the extension point for persistent backends.

Commodity algorithm note
------------------------
Version diffing uses a line-by-line comparison (``difflib.ndiff``) to
produce a human-readable unified-diff-style output.  No proprietary
similarity scoring is used.

Classes
-------
PromptRegistryError
    Base exception for this module.
PromptNotFoundError
    Raised when a prompt name is not found in the registry.
PromptVersionNotFoundError
    Raised when a specific version number is not found.
PromptVersion
    Immutable record of a stored prompt version.
PromptTemplate
    A resolved prompt ready for rendering.
PromptDiff
    The diff result between two prompt versions.
PromptRegistry
    Central registry managing prompt templates.
"""
from __future__ import annotations

import difflib
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromptRegistryError(Exception):
    """Base exception for all prompt registry errors."""


class PromptNotFoundError(PromptRegistryError):
    """Raised when the requested prompt name does not exist in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Prompt '{name}' not found in registry.")


class PromptVersionNotFoundError(PromptRegistryError):
    """Raised when the requested version number does not exist for a prompt."""

    def __init__(self, name: str, version: int) -> None:
        self.name = name
        self.version = version
        super().__init__(f"Version {version} of prompt '{name}' not found.")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptVersion:
    """Immutable record representing a single stored version of a prompt.

    Attributes
    ----------
    name:
        The logical name of the prompt (e.g. ``"summarize"``).
    version:
        Auto-incrementing integer version number, starting at 1.
    template:
        Raw template string with ``{variable}`` placeholders.
    variables:
        List of variable names expected by the template.
    metadata:
        Arbitrary key-value pairs attached at registration time.
    created_at:
        UTC timestamp at which this version was registered.
    """

    name: str
    version: int
    template: str
    variables: list[str]
    metadata: dict[str, object]
    created_at: datetime


@dataclass(frozen=True)
class PromptTemplate:
    """A resolved prompt template ready for rendering.

    Attributes
    ----------
    name:
        The logical name of the prompt.
    version:
        The version number this template was retrieved from.
    template:
        Raw template string.
    variables:
        Expected variable names.
    """

    name: str
    version: int
    template: str
    variables: list[str]

    def render(self, **variables: object) -> str:
        """Render the template by substituting keyword arguments.

        Parameters
        ----------
        **variables:
            Values for each ``{variable}`` placeholder in the template.

        Returns
        -------
        str
            The fully rendered prompt string.

        Raises
        ------
        KeyError
            If a required variable is missing from *variables*.
        """
        return self.template.format(**variables)


@dataclass
class PromptDiff:
    """The result of diffing two versions of the same prompt.

    Attributes
    ----------
    name:
        Prompt name.
    version_a:
        Earlier version number.
    version_b:
        Later version number.
    template_a:
        Raw template string of version *a*.
    template_b:
        Raw template string of version *b*.
    diff_lines:
        Line-by-line unified diff output (``difflib.ndiff`` format).
        Lines starting with ``-`` were removed, ``+`` were added,
        and ``?`` are hints.  Lines with neither prefix are unchanged.
    added_variables:
        Variable names present in *b* but not *a*.
    removed_variables:
        Variable names present in *a* but not *b*.
    is_identical:
        True when both templates are byte-for-byte identical.
    """

    name: str
    version_a: int
    version_b: int
    template_a: str
    template_b: str
    diff_lines: list[str]
    added_variables: list[str]
    removed_variables: list[str]

    @property
    def is_identical(self) -> bool:
        """Return True when both template strings are identical."""
        return self.template_a == self.template_b


# ---------------------------------------------------------------------------
# Backend protocol (extension point)
# ---------------------------------------------------------------------------


class PromptStorageBackend:
    """Extension point for persistent prompt storage.

    The in-memory default stores all data in ``_store``.  Replace this
    class with a database-backed implementation by subclassing and
    overriding all methods.

    Attributes
    ----------
    _store:
        Mapping of prompt name to list of PromptVersion (oldest first).
    """

    def __init__(self) -> None:
        self._store: dict[str, list[PromptVersion]] = {}

    def save(self, prompt_version: PromptVersion) -> None:
        """Persist a new prompt version."""
        if prompt_version.name not in self._store:
            self._store[prompt_version.name] = []
        self._store[prompt_version.name].append(prompt_version)

    def get_all(self, name: str) -> list[PromptVersion]:
        """Return all versions for a given prompt name, oldest first."""
        return list(self._store.get(name, []))

    def list_names(self) -> list[str]:
        """Return all registered prompt names, sorted."""
        return sorted(self._store.keys())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _extract_template_variables(template: str) -> list[str]:
    """Extract all ``{variable}`` placeholder names from a template string.

    Uses ``string.Formatter().parse`` so it handles the same syntax as
    ``str.format``.  Each unique field name is returned once, in order of
    first appearance.

    Parameters
    ----------
    template:
        A format string, e.g. ``"Hello, {name}!"``.

    Returns
    -------
    list[str]
        Unique field names, preserving order of first appearance.
    """
    seen: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is not None and field_name not in seen:
            seen.append(field_name)
    return seen


class PromptRegistry:
    """Central registry for prompt templates with version tracking.

    All prompts are identified by a string *name*.  Each call to
    :meth:`register` creates a new immutable :class:`PromptVersion` whose
    version number is auto-incremented per prompt.

    Parameters
    ----------
    backend:
        Storage backend.  Defaults to :class:`PromptStorageBackend` (in-memory).

    Example
    -------
    ::

        registry = PromptRegistry()
        registry.register("greet", "Hello, {name}!", variables=["name"])
        registry.register("greet", "Hi there, {name}!")
        text = registry.render("greet", name="Alice")
        diff = registry.diff("greet", 1, 2)
    """

    def __init__(
        self,
        backend: PromptStorageBackend | None = None,
    ) -> None:
        self._backend: PromptStorageBackend = (
            backend if backend is not None else PromptStorageBackend()
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        template: str,
        variables: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PromptVersion:
        """Register a new version of a prompt template.

        If *variables* is not supplied, the registry will auto-detect
        placeholders by parsing the template string.

        Parameters
        ----------
        name:
            Logical name for the prompt (e.g. ``"summarize"``).
        template:
            Raw template string with ``{variable}`` placeholders.
        variables:
            Explicit list of expected variable names.  When ``None``,
            inferred automatically from the template.
        metadata:
            Arbitrary key-value pairs to store alongside the version.

        Returns
        -------
        PromptVersion
            The newly created version record.
        """
        existing = self._backend.get_all(name)
        next_version = len(existing) + 1

        resolved_variables: list[str] = (
            variables if variables is not None else _extract_template_variables(template)
        )

        prompt_version = PromptVersion(
            name=name,
            version=next_version,
            template=template,
            variables=resolved_variables,
            metadata=dict(metadata) if metadata else {},
            created_at=datetime.now(tz=timezone.utc),
        )
        self._backend.save(prompt_version)
        return prompt_version

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(
        self,
        name: str,
        version: int | None = None,
    ) -> PromptTemplate:
        """Retrieve a prompt template by name and optional version.

        Parameters
        ----------
        name:
            Prompt name.
        version:
            Specific version number to retrieve.  When ``None``, the
            latest (highest-numbered) version is returned.

        Returns
        -------
        PromptTemplate
            The resolved template ready for rendering.

        Raises
        ------
        PromptNotFoundError
            If *name* is not in the registry.
        PromptVersionNotFoundError
            If *version* is specified but does not exist.
        """
        all_versions = self._backend.get_all(name)
        if not all_versions:
            raise PromptNotFoundError(name)

        if version is None:
            selected = all_versions[-1]
        else:
            matches = [v for v in all_versions if v.version == version]
            if not matches:
                raise PromptVersionNotFoundError(name, version)
            selected = matches[0]

        return PromptTemplate(
            name=selected.name,
            version=selected.version,
            template=selected.template,
            variables=selected.variables,
        )

    def list_versions(self, name: str) -> list[PromptVersion]:
        """Return all stored versions for a prompt, oldest first.

        Parameters
        ----------
        name:
            Prompt name.

        Returns
        -------
        list[PromptVersion]
            All versions in ascending version-number order.

        Raises
        ------
        PromptNotFoundError
            If *name* is not in the registry.
        """
        all_versions = self._backend.get_all(name)
        if not all_versions:
            raise PromptNotFoundError(name)
        return list(all_versions)

    def list_prompts(self) -> list[str]:
        """Return all registered prompt names.

        Returns
        -------
        list[str]
            Sorted list of known prompt names.
        """
        return self._backend.list_names()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        prompt_name: str,
        version: int | None = None,
        **variables: object,
    ) -> str:
        """Render a prompt template by substituting variables.

        Parameters
        ----------
        prompt_name:
            Prompt name (named ``prompt_name`` to avoid colliding with a
            template variable also called ``name``).
        version:
            Version to render.  Defaults to the latest.
        **variables:
            Values for each ``{variable}`` placeholder.

        Returns
        -------
        str
            The rendered prompt string.

        Raises
        ------
        PromptNotFoundError
            If *prompt_name* is not in the registry.
        PromptVersionNotFoundError
            If *version* is not found.
        KeyError
            If a required template variable is missing from *variables*.
        """
        template = self.get(prompt_name, version)
        return template.render(**variables)

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    def diff(
        self,
        name: str,
        version_a: int,
        version_b: int,
    ) -> PromptDiff:
        """Compare two versions of the same prompt.

        Uses ``difflib.ndiff`` for line-level comparison.

        Parameters
        ----------
        name:
            Prompt name.
        version_a:
            Earlier version to compare from.
        version_b:
            Later version to compare to.

        Returns
        -------
        PromptDiff
            A structured diff result including line changes and variable
            additions/removals.

        Raises
        ------
        PromptNotFoundError
            If *name* is not in the registry.
        PromptVersionNotFoundError
            If either version is not found.
        """
        pv_a = self._get_version_record(name, version_a)
        pv_b = self._get_version_record(name, version_b)

        lines_a = pv_a.template.splitlines(keepends=True)
        lines_b = pv_b.template.splitlines(keepends=True)
        diff_lines = list(difflib.ndiff(lines_a, lines_b))

        vars_a = set(pv_a.variables)
        vars_b = set(pv_b.variables)

        return PromptDiff(
            name=name,
            version_a=version_a,
            version_b=version_b,
            template_a=pv_a.template,
            template_b=pv_b.template,
            diff_lines=diff_lines,
            added_variables=sorted(vars_b - vars_a),
            removed_variables=sorted(vars_a - vars_b),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version_record(self, name: str, version: int) -> PromptVersion:
        """Retrieve a raw PromptVersion record, raising descriptive errors."""
        all_versions = self._backend.get_all(name)
        if not all_versions:
            raise PromptNotFoundError(name)
        matches = [v for v in all_versions if v.version == version]
        if not matches:
            raise PromptVersionNotFoundError(name, version)
        return matches[0]
