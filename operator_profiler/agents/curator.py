"""
MemoryCuratorAgent — deduplicates and compacts OptimizationMemory entries.

Identifies three classes of removable entries:
  Dominated    Same op pattern + bottleneck, but strictly lower speedup than
               another entry with the same rewrite op types.
  Near-duplicate  Jaccard similarity above threshold with identical rewrite op
               type sets — keeps the higher-speedup representative.
  Stale        Speedup ≤ 1.0× (no actual improvement was measured).

The agent is conservative: when in doubt, it keeps entries.

Integration
-----------
OptimizationMemory.compact(curator_agent) calls agent.curate(entries), applies
the CurationResult by dropping entries_to_remove from the store, and persists
atomically.  The method returns a CurationResult so callers can log reasoning.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator_profiler.planner.schema import MemoryEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory management agent for a GPU optimization system. You maintain
a store of successful rewrite entries. Your job is to keep the store high-quality
and non-redundant so retrieval remains effective.

Remove entries that are:
- Dominated: same op pattern + bottleneck, lower speedup than another entry
  that used the same rewrite op types.
- Near-duplicate: Jaccard similarity ≥ 0.85 and identical rewrite op types;
  keep only the highest-speedup representative.
- Stale: speedup ≤ 1.0× (measurement noise, no actual gain).

Be conservative — only remove entries that are clearly redundant or harmful.
When two entries have similar speedup, keep both (diverse retrieval pool is
valuable). Only remove when one strictly dominates the other.

Respond only by calling the curate_memory tool."""

_CURATE_TOOL = {
    "name": "curate_memory",
    "description": "Decide which memory entries to keep and which to remove.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entries_to_keep": {
                "type": "array",
                "items": {"type": "string"},
                "description": "entry_id values to retain in the store.",
            },
            "entries_to_remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "entry_id values to remove. Only include entries that are clearly "
                    "dominated, near-duplicate, or stale."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Two to four sentences explaining curation decisions. "
                    "For each removed entry, name the dominant entry that supersedes it."
                ),
            },
        },
        "required": ["entries_to_keep", "entries_to_remove", "reasoning"],
    },
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CurationResult:
    entries_to_keep: list[str]
    entries_to_remove: list[str]
    reasoning: str
    removed_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.removed_count = len(self.entries_to_remove)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MemoryCuratorAgent:
    """
    Deduplicates and compacts OptimizationMemory entries via an LLM.

    Parameters
    ----------
    model:
        Anthropic model string.
    api_key:
        Optional API key; falls back to ANTHROPIC_API_KEY env var.
    jaccard_threshold:
        Entries with Jaccard similarity above this threshold are flagged as
        near-duplicate candidates in the prompt.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        jaccard_threshold: float = 0.85,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._jaccard_threshold = jaccard_threshold
        self._client = self._build_client()

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for MemoryCuratorAgent. "
                "Install with: pip install anthropic"
            ) from exc
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate(self, entries: "list[MemoryEntry]") -> CurationResult:
        """
        Review ``entries`` and return a CurationResult.

        Parameters
        ----------
        entries:
            All current entries in the memory store.

        Returns
        -------
        CurationResult
            Falls back to keeping all entries on API or parse error.
        """
        if len(entries) <= 1:
            return CurationResult(
                entries_to_keep=[e.entry_id for e in entries],
                entries_to_remove=[],
                reasoning="Store has ≤1 entries; no curation needed.",
            )

        user_message = self._build_message(entries)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
                tools=[_CURATE_TOOL],
                tool_choice={"type": "tool", "name": "curate_memory"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            log.warning(
                "MemoryCuratorAgent API call failed: %s — keeping all entries", exc
            )
            return CurationResult(
                entries_to_keep=[e.entry_id for e in entries],
                entries_to_remove=[],
                reasoning=f"API call failed: {exc}",
            )

        return self._parse_response(response, entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 1.0

    def _build_message(self, entries: "list[MemoryEntry]") -> str:
        # Flag near-duplicate pairs above threshold
        near_dupes: list[str] = []
        for i, ei in enumerate(entries):
            for j, ej in enumerate(entries):
                if j <= i:
                    continue
                sim = self._jaccard(
                    ei.graph_pattern.op_sequence,
                    ej.graph_pattern.op_sequence,
                )
                if sim >= self._jaccard_threshold:
                    ei_ops = sorted({o.op for o in ei.rewrite_plan.ops})
                    ej_ops = sorted({o.op for o in ej.rewrite_plan.ops})
                    same_ops = ei_ops == ej_ops
                    near_dupes.append(
                        f"  {ei.entry_id} ↔ {ej.entry_id}: "
                        f"Jaccard={sim:.2f}  "
                        f"speedup {ei.speedup:.3f}× vs {ej.speedup:.3f}×  "
                        f"same_rewrite_types={same_ops}"
                    )

        # Compact per-entry summaries
        summaries = []
        for e in entries:
            op_types = sorted({o.op for o in e.rewrite_plan.ops})
            summaries.append({
                "entry_id": e.entry_id,
                "op_sequence": e.graph_pattern.op_sequence,
                "bottleneck": e.bottleneck,
                "rewrite_op_types": op_types,
                "speedup": round(e.speedup, 3),
                "model": e.model_name,
                "created_at": e.created_at,
            })

        lines = [
            f"## Memory Store ({len(entries)} entries)",
            json.dumps(summaries, indent=2),
        ]
        if near_dupes:
            lines += [
                "",
                f"## Near-Duplicate Pairs (Jaccard ≥ {self._jaccard_threshold}):",
            ]
            lines.extend(near_dupes)

        lines += [
            "",
            "Decide which entries to keep and which to remove. "
            "Be conservative — only remove clearly dominated or redundant entries.",
        ]
        return "\n".join(lines)

    def _parse_response(
        self,
        response,
        entries: "list[MemoryEntry]",
    ) -> CurationResult:
        all_ids = {e.entry_id for e in entries}

        for block in response.content:
            if block.type == "tool_use" and block.name == "curate_memory":
                try:
                    inp = block.input
                    to_keep = [eid for eid in inp.get("entries_to_keep", []) if eid in all_ids]
                    to_remove = [eid for eid in inp.get("entries_to_remove", []) if eid in all_ids]

                    # Safety: any ID not explicitly removed is implicitly kept
                    removed_set = set(to_remove)
                    kept_set = set(to_keep)
                    for eid in all_ids:
                        if eid not in removed_set and eid not in kept_set:
                            to_keep.append(eid)

                    return CurationResult(
                        entries_to_keep=to_keep,
                        entries_to_remove=to_remove,
                        reasoning=inp.get("reasoning", ""),
                    )
                except Exception as exc:
                    log.warning("MemoryCuratorAgent parse error: %s", exc)
                    break

        log.warning("MemoryCuratorAgent: no tool_use block — keeping all entries")
        return CurationResult(
            entries_to_keep=[e.entry_id for e in entries],
            entries_to_remove=[],
            reasoning="No tool_use block returned; kept all entries as a safety fallback.",
        )
