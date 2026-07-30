"""
Activity detection: segment keypoint sequences into active regions.
Cuts out zero-frame segments (hand loss) so CTC trains on meaningful frames only.
"""
import numpy as np


def segment_active_regions(kp_sequence, min_active_frames=5, gap_frames=3):
    """
    kp_sequence: (T, 84) numpy array
    Returns: list of (start, end) segment indices

    Algorithm:
    - Find runs of active frames (abs sum > 1e-6)
    - Merge adjacent active runs separated by gaps ≤ gap_frames
    - Keep segments where total active frames ≥ min_active_frames
    - The gap frames between merged runs are INCLUDED in the segment
    """
    is_active = np.abs(kp_sequence).sum(axis=1) > 1e-6
    n = len(is_active)

    # Collect active runs: (start, end)
    active_runs = []
    i = 0
    while i < n:
        if is_active[i]:
            start = i
            while i < n and is_active[i]:
                i += 1
            active_runs.append((start, i))
        else:
            i += 1

    if not active_runs:
        return []

    # Merge active runs separated by gaps ≤ gap_frames
    segments = []
    cur_start, cur_end = active_runs[0]
    cur_active_frames = cur_end - cur_start

    for next_start, next_end in active_runs[1:]:
        gap = next_start - cur_end
        if gap <= gap_frames:
            cur_end = next_end
            cur_active_frames += (next_end - next_start)
        else:
            if cur_active_frames >= min_active_frames:
                segments.append((cur_start, cur_end))
            cur_start, cur_end = next_start, next_end
            cur_active_frames = next_end - next_start

    if cur_active_frames >= min_active_frames:
        segments.append((cur_start, cur_end))

    return segments


def extract_active_frames(kp_sequence, segments, separator_frames=4):
    """
    Concatenate active segments with zero-frame separators between them.
    Returns original sequence if no segments found.
    """
    if len(segments) == 0:
        return kp_sequence

    parts = []
    sep = np.zeros((separator_frames, kp_sequence.shape[1]), dtype=kp_sequence.dtype)
    for i, (start, end) in enumerate(segments):
        if i > 0:
            parts.append(sep)
        parts.append(kp_sequence[start:end])

    return np.concatenate(parts, axis=0)
