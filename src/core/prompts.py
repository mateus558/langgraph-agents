"""Reusable prompt abstractions built on top of :class:`PromptProtocol`.

This module keeps the abstraction intentionally lightweight: prompts are
rendered with ``str.format`` and return LangChain-compatible message objects.
It centralizes common validation/error handling so callers can focus on
supplying variables instead of re-creating prompt plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core.contracts import PromptProtocol


class PromptRenderError(ValueError):
    """Raised when a prompt cannot be rendered due to validation issues."""


_formatter = Formatter()


def _required_fields(template: str) -> set[str]:
    """Return placeholder field names referenced by the template."""
    fields: set[str] = set()
    for _, field_name, _, _ in _formatter.parse(template):
        if field_name:
            # Support dotted names (e.g. user.name) by taking the root key
            field_root = field_name.split(".", 1)[0]
            if field_root:
                fields.add(field_root)
    return fields


def _merge_kwargs(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge defaults with overrides (overrides win)."""
    if not defaults:
        return dict(overrides)
    merged = dict(defaults)
    merged.update(overrides)
    return merged


@dataclass(frozen=True)
class ChatPrompt(PromptProtocol):
    """Concrete chat prompt implementation built on ``str.format`` templates.

    Args:
        id: Stable identifier used for logging/analytics.
        version: Optional semantic version string for the prompt.
        system_template: Optional template rendered into a system message.
        user_template: Template rendered into the human message.
        defaults: Default values supplied for formatting placeholders.
        metadata: Arbitrary metadata that downstream systems might log.

    The class validates missing placeholders before attempting to render in
    order to surface more informative errors. ``messages`` always returns at
    least one :class:`HumanMessage`.
    """

    id: str
    version: str = "1"
    system_template: str | None = None
    user_template: str = ""
    defaults: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.user_template

    def _validate(self, template: str | None, values: Mapping[str, Any]) -> None:
        if not template:
            return
        missing = _required_fields(template) - set(values.keys())
        if missing:
            raise PromptRenderError(
                f"Prompt '{self.id}' missing values for: {', '.join(sorted(missing))}"
            )

    def _render_template(self, template: str | None, values: Mapping[str, Any]) -> str | None:
        if template is None:
            return None
        self._validate(template, values)
        try:
            return template.format(**values)
        except KeyError as exc:
            raise PromptRenderError(f"Prompt '{self.id}' missing key: {exc}") from exc

    def format(self, **kwargs: Any) -> str:
        values = _merge_kwargs(self.defaults, kwargs)
        rendered = self._render_template(self.user_template, values)
        if rendered is None:
            raise PromptRenderError(f"Prompt '{self.id}' has no user_template content.")
        return rendered

    def format_system(self, **kwargs: Any) -> str | None:
        values = _merge_kwargs(self.defaults, kwargs)
        return self._render_template(self.system_template, values)

    def messages(self, **kwargs: Any) -> list[BaseMessage]:
        values = _merge_kwargs(self.defaults, kwargs)
        system_text = self._render_template(self.system_template, values)
        human_text = self._render_template(self.user_template, values)
        if human_text is None:
            raise PromptRenderError(f"Prompt '{self.id}' produced no human message.")
        messages: list[BaseMessage] = []
        if system_text:
            messages.append(SystemMessage(content=system_text))
        messages.append(HumanMessage(content=human_text))
        return messages

    def required_fields(self) -> set[str]:
        """Return the set of field names required by the system/user templates."""
        fields: set[str] = set()
        if self.system_template:
            fields |= _required_fields(self.system_template)
        if self.user_template:
            fields |= _required_fields(self.user_template)
        return fields

    def describe(self) -> dict[str, Any]:
        """Return a structured description for observability/debugging."""
        return {
            "id": self.id,
            "version": self.version,
            "metadata": dict(self.metadata),
            "required_fields": sorted(self.required_fields()),
        }


__all__ = ["ChatPrompt", "PromptRenderError"]
