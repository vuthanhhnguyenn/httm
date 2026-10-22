"""Signal hierarchy: one root cause produces one event.

When a root signal (for example ``PERSON_MISSING``) is pending or active, the
signals it explains (face not visible, head turns, …) do not open events of
their own. They are recorded on the root event as ``supporting_signals`` and
add no risk. A root that is merely a candidate already holds its consequences
back, so a short absence cannot leak a "face not visible" event first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..types import EventType
from .signals import RuleSignal, SignalState

SUPPORTING_KEY = "supporting_signals"


class SignalArbiter:
    def __init__(self, hierarchy: Mapping[EventType, Sequence[EventType]]) -> None:
        self.hierarchy = {EventType(root): frozenset(EventType(item) for item in children)
                          for root, children in hierarchy.items()}
        self._supporting: dict[EventType, set[EventType]] = {}
        # Explained signals stay held until their own condition resets, so a long absence
        # cannot release an already-satisfied "face not visible" the moment the person returns.
        self._held: set[EventType] = set()

    def apply(self, signals: Sequence[RuleSignal]) -> list[RuleSignal]:
        engaged = {signal.event_type for signal in signals
                   if signal.event_type in self.hierarchy
                   and signal.state in {SignalState.ACTIVE, SignalState.CANDIDATE}}
        for root in tuple(self._supporting):
            if root not in engaged:
                del self._supporting[root]
        # A rule that emits nothing this frame is inactive, so its condition has reset.
        self._held &= {signal.event_type for signal in signals}
        explained_by: dict[EventType, list[EventType]] = {}
        for root in engaged:
            for child in self.hierarchy[root]:
                explained_by.setdefault(child, []).append(root)

        arbitrated: list[RuleSignal] = []
        for signal in signals:
            roots = explained_by.get(signal.event_type)
            if not roots:
                if signal.event_type in self._held and signal.state is SignalState.ACTIVE:
                    arbitrated.append(replace(signal, state=SignalState.INACTIVE))
                    continue
                self._held.discard(signal.event_type)
                arbitrated.append(signal)
                continue
            self._held.add(signal.event_type)
            if signal.state is SignalState.ACTIVE:
                for root in roots:
                    self._supporting.setdefault(root, set()).add(signal.event_type)
            # INACTIVE closes an event of this type that opened before the root engaged.
            arbitrated.append(replace(signal, state=SignalState.INACTIVE))

        result: list[RuleSignal] = []
        for signal in arbitrated:
            supporting = self._supporting.get(signal.event_type)
            if supporting and signal.state is SignalState.ACTIVE:
                names = ",".join(sorted(item.value for item in supporting))
                signal = replace(signal, metrics={**signal.metrics, SUPPORTING_KEY: names})
            result.append(signal)
        return result


def validate_hierarchy(hierarchy: Mapping[EventType, Sequence[EventType]]) -> None:
    """Reject self-references and cycles so suppression always has a root."""

    graph = {EventType(root): {EventType(child) for child in children} for root, children in hierarchy.items()}
    visiting: set[EventType] = set()
    done: set[EventType] = set()

    def visit(node: EventType, path: tuple[EventType, ...]) -> None:
        if node in done:
            return
        if node in visiting:
            cycle = " → ".join(item.value for item in (*path, node))
            raise ValueError(f"signal hierarchy has a cycle: {cycle}")
        visiting.add(node)
        for child in graph.get(node, ()):
            visit(child, (*path, node))
        visiting.discard(node)
        done.add(node)

    for root in graph:
        visit(root, ())


__all__ = ["SUPPORTING_KEY", "SignalArbiter", "validate_hierarchy"]
