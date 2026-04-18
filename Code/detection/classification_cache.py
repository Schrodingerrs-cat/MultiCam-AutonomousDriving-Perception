"""
Classification Cache — Eliminates Redundant VLM / Zero-Shot Calls
=================================================================

Problem:
    The pipeline was calling Qwen VQA (~1-3s) and SigLIP (~200ms) on
    EVERY detection on EVERY frame, even for the same tracked object.
    With 10 objects × 5 classification paths = ~50 seconds/frame.

Solution:
    Cache classification results keyed by (track_id, classifier_name).
    Only re-classify when:
      a) track_id is new (never seen before)
      b) N frames have elapsed since last classification (cadence)
      c) previous confidence was low (retry with fresh crop)

All VLM-using modules should call cache.get_or_classify() instead of
running inference directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CachedResult:
    """One cached classification result."""
    result: dict               # the classification output (arbitrary dict)
    frame_idx: int             # frame when this was computed
    confidence: str = "high"   # "high" | "medium" | "low"
    hit_count: int = 0         # how many times this cache entry was reused


class ClassificationCache:
    """
    Per-track, per-classifier cache with cadence control.

    Parameters:
        cadence_high:   frames between re-classifications for high-confidence results
        cadence_low:    frames between re-classifications for low-confidence results
        max_stale:      frames after which a cached result is discarded entirely
    """

    def __init__(self,
                 cadence_high: int = 15,
                 cadence_low: int = 5,
                 max_stale: int = 90):
        self.cadence_high = cadence_high
        self.cadence_low = cadence_low
        self.max_stale = max_stale
        # (track_id, classifier_name) → CachedResult
        self._cache: dict[tuple[int, str], CachedResult] = {}

    def get_or_classify(
        self,
        track_id: int,
        classifier_name: str,
        frame_idx: int,
        classify_fn: Callable[[], Optional[dict]],
        confidence_key: str = "confidence",
    ) -> Optional[dict]:
        """
        Return a cached result if fresh enough, or call classify_fn() and cache it.

        Args:
            track_id:        stable track ID for the object
            classifier_name: e.g. "sign_qwen", "tl_siglip", "small_obj_siglip"
            frame_idx:       current frame number
            classify_fn:     zero-arg callable that runs the actual classification.
                             Called ONLY if cache miss. May return None on failure.
            confidence_key:  key in the result dict that holds confidence level

        Returns:
            The classification result dict, or None if classification failed
            and no cached result exists.
        """
        key = (track_id, classifier_name)
        cached = self._cache.get(key)

        if cached is not None:
            age = frame_idx - cached.frame_idx

            # Discard if too old
            if age > self.max_stale:
                del self._cache[key]
                cached = None
            else:
                # Check cadence
                conf = cached.confidence
                cadence = self.cadence_low if conf == "low" else self.cadence_high
                if age < cadence:
                    # Cache hit — return without re-classifying
                    cached.hit_count += 1
                    return cached.result

        # Cache miss or stale — run classifier
        result = classify_fn()

        if result is not None:
            conf = str(result.get(confidence_key, "high")).lower()
            if conf not in ("high", "medium", "low"):
                conf = "medium"
            self._cache[key] = CachedResult(
                result=result,
                frame_idx=frame_idx,
                confidence=conf,
            )
            return result

        # Classification failed — return stale cache if available
        if cached is not None:
            cached.hit_count += 1
            return cached.result

        return None

    def get_cached(self, track_id: int, classifier_name: str) -> Optional[dict]:
        """Return cached result without triggering re-classification."""
        cached = self._cache.get((track_id, classifier_name))
        return cached.result if cached is not None else None

    def put(self, track_id: int, classifier_name: str,
            result: dict, frame_idx: int,
            confidence: str = "high"):
        """Manually insert a result into the cache."""
        self._cache[(track_id, classifier_name)] = CachedResult(
            result=result, frame_idx=frame_idx, confidence=confidence,
        )

    def prune(self, active_track_ids: set):
        """Remove entries for tracks no longer active."""
        stale_keys = [k for k in self._cache if k[0] not in active_track_ids]
        for k in stale_keys:
            del self._cache[k]

    def reset(self):
        """Clear all state (call between sequences)."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics for logging."""
        total = len(self._cache)
        total_hits = sum(c.hit_count for c in self._cache.values())
        return {"entries": total, "total_cache_hits": total_hits}
