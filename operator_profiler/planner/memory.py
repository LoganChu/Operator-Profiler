"""
OptimizationMemory — a JSON-RAG store for successful graph rewrites.

Each entry is a ``MemoryEntry`` triplet:
    (GraphPattern, bottleneck_classification, RewritePlan, speedup)

Retrieval uses Jaccard similarity on the operator-name token sets, filtered
by bottleneck class.  Persistence is atomic: write to a `.tmp` file then
``os.replace()`` to prevent corruption on crash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from operator_profiler.rewriter.dsl import RewritePlan
from operator_profiler.schema.profile import OperatorAttributedProfile
from operator_profiler.planner.schema import (
    GraphPattern,
    MemoryEntry,
    OptMemoryStore,
    SearchCandidate,
)

logger = logging.getLogger(__name__)


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two op-sequence token sets."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _make_pattern_hash(op_sequence: list[str]) -> str:
    """SHA-256 of the sorted, pipe-joined op sequence (order-independent)."""
    canonical = "|".join(sorted(op_sequence))
    return hashlib.sha256(canonical.encode()).hexdigest()


class OptimizationMemory:
    """
    Persistent JSON store of successful rewrite triplets.

    Usage
    -----
    mem = OptimizationMemory("opt_memory.json")
    candidates = mem.search(pattern, "memory_bound", top_k=3)
    entry = mem.curate(profile, plan, speedup=1.18)
    """

    def __init__(self, path: str | Path = "opt_memory.json") -> None:
        self._path = Path(path)
        self._store = OptMemoryStore()
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read and deserialize the store from disk."""
        raw = self._path.read_text(encoding="utf-8")
        self._store = OptMemoryStore.model_validate_json(raw)
        logger.debug("Loaded %d memory entries from %s", len(self._store.entries), self._path)

    def save_store(self) -> None:
        """Atomically write the store to disk."""
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            self._store.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)
        logger.debug("Saved %d memory entries to %s", len(self._store.entries), self._path)

    # ------------------------------------------------------------------
    # Pattern extraction
    # ------------------------------------------------------------------

    def extract_pattern(self, profile: OperatorAttributedProfile) -> GraphPattern:
        """
        Build a ``GraphPattern`` from an ``OperatorAttributedProfile``.

        ``op_sequence`` preserves call order; ``pattern_hash`` is
        order-independent (sorted SHA-256) so that equivalent graphs hash
        identically regardless of insertion order.
        """
        op_sequence = [op.operator_name for op in profile.operators]
        input_shapes: dict[str, list[int]] = {}
        for op in profile.operators:
            if op.source_location is not None:
                pass  # shapes come from capture_metadata if available
        return GraphPattern(
            op_sequence=op_sequence,
            pattern_hash=_make_pattern_hash(op_sequence),
            input_shapes=input_shapes,
        )

    # ------------------------------------------------------------------
    # Search (JSON-RAG retrieval)
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: GraphPattern,
        bottleneck: str,
        top_k: int = 3,
    ) -> list[SearchCandidate]:
        """
        Retrieve the top-K memory entries most similar to ``pattern`` that
        share the same ``bottleneck`` classification.

        Ranking: Jaccard similarity descending; ties broken by speedup
        descending.
        """
        candidates: list[SearchCandidate] = []
        for entry in self._store.entries:
            if entry.bottleneck != bottleneck:
                continue
            sim = _jaccard(pattern.op_sequence, entry.graph_pattern.op_sequence)
            candidates.append(SearchCandidate(entry=entry, similarity=sim))

        candidates.sort(key=lambda c: (c.similarity, c.entry.speedup), reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Curation
    # ------------------------------------------------------------------

    def curate(
        self,
        profile: OperatorAttributedProfile,
        plan: RewritePlan,
        speedup: float,
        speedup_threshold: float = 1.05,
    ) -> MemoryEntry | None:
        """
        Save a new entry if ``speedup > speedup_threshold``.

        Returns the saved ``MemoryEntry`` or ``None`` if below threshold.
        """
        if speedup <= speedup_threshold:
            return None

        bottleneck = _worst_bottleneck(profile)
        pattern = self.extract_pattern(profile)
        entry = MemoryEntry(
            entry_id=uuid.uuid4().hex,
            graph_pattern=pattern,
            bottleneck=bottleneck,
            rewrite_plan=plan,
            speedup=speedup,
            profile_id=profile.schema_version,
            model_name=profile.capture_metadata.model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store.entries.append(entry)
        self.save_store()
        logger.info(
            "Curated memory entry %s (speedup=%.3fx, bottleneck=%s)",
            entry.entry_id,
            speedup,
            bottleneck,
        )
        return entry

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store.entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._store.entries)


# ---------------------------------------------------------------------------
# Module-level helper (also used by loop.py)
# ---------------------------------------------------------------------------

def _worst_bottleneck(profile: OperatorAttributedProfile) -> str:
    """
    Return the ``bottleneck_classification`` of the operator with the
    highest ``total_duration_ns``.  Falls back to ``"unknown"``.
    """
    worst_op = None
    worst_duration = -1
    for op in profile.operators:
        if op.aggregated is not None and op.aggregated.total_duration_ns > worst_duration:
            worst_duration = op.aggregated.total_duration_ns
            worst_op = op
    if worst_op is None or worst_op.aggregated is None:
        return "unknown"
    return worst_op.aggregated.bottleneck_classification
