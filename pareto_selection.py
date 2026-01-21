"""
Pareto-based candidate selection for Bloat Axis GEPA.

Implements Algorithm 3 (SelectCandidate) from the GEPA paper.
Uses Pareto illumination to maintain diversity and balance exploration/exploitation.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

try:
    from .gepa_types import CandidatePool
except ImportError:
    from gepa_types import CandidatePool

logger = logging.getLogger(__name__)


def select_candidate(
    pool: CandidatePool,
    instance_indices: List[int],
) -> int:
    """
    Pareto-based candidate selection (Algorithm 3).
    
    1. For each instance, find the candidates with maximum score
    2. Collect all candidates that win on at least one instance
    3. Remove dominated candidates (Pareto filtering)
    4. Sample from non-dominated candidates proportionally to win frequency
    
    Args:
        pool: CandidatePool with scores
        instance_indices: Indices of instances in D_pareto
        
    Returns:
        Index of selected candidate in pool.candidates
    """
    if len(pool.candidates) == 0:
        raise ValueError("Cannot select from empty candidate pool")
    
    if len(pool.candidates) == 1:
        return 0
    
    # Step 1: Find winners per instance
    # P*[i] = candidates that achieve max score on instance i
    winners_per_instance: Dict[int, Set[int]] = {}
    max_score_per_instance: Dict[int, float] = {}
    
    for i in instance_indices:
        max_score = float('-inf')
        winners = set()
        
        for c_idx in range(len(pool.candidates)):
            score = pool.get_score(c_idx, i)
            if score is None:
                continue
            
            if score > max_score:
                max_score = score
                winners = {c_idx}
            elif score == max_score:
                winners.add(c_idx)
        
        winners_per_instance[i] = winners
        max_score_per_instance[i] = max_score
    
    # Step 2: Collect all candidates that win somewhere
    # C = unique candidates in union of P*[i]
    all_winners: Set[int] = set()
    for winners in winners_per_instance.values():
        all_winners |= winners
    
    if not all_winners:
        # No valid scores - return random candidate
        logger.warning("No candidates with valid scores, selecting randomly")
        return random.randint(0, len(pool.candidates) - 1)
    
    # Step 3: Remove dominated candidates
    # D = candidates dominated by another in C
    dominated = find_dominated_candidates(
        list(all_winners),
        pool,
        instance_indices,
    )
    
    # Non-dominated candidates
    non_dominated = all_winners - dominated
    
    if not non_dominated:
        # All candidates dominated each other (ties) - use all winners
        non_dominated = all_winners
    
    # Step 4: Compute frequency and sample
    # f[Φ] = number of instances where Φ wins (after removing dominated)
    frequency: Dict[int, int] = defaultdict(int)
    
    for i, winners in winners_per_instance.items():
        # Remove dominated from winners
        valid_winners = winners - dominated
        for c_idx in valid_winners:
            frequency[c_idx] += 1
    
    # Sample proportionally to frequency
    candidates_list = list(non_dominated)
    weights = [frequency[c] for c in candidates_list]
    
    # Handle case where all weights are 0
    if sum(weights) == 0:
        weights = [1] * len(candidates_list)
    
    selected = random.choices(candidates_list, weights=weights, k=1)[0]
    
    logger.debug(
        f"Pareto selection: {len(all_winners)} winners, "
        f"{len(dominated)} dominated, {len(non_dominated)} non-dominated, "
        f"selected candidate {selected} (freq={frequency[selected]})"
    )
    
    return selected


def find_dominated_candidates(
    candidates: List[int],
    pool: CandidatePool,
    instance_indices: List[int],
) -> Set[int]:
    """
    Find candidates that are dominated by another candidate.
    
    Candidate A dominates B if:
    - A scores >= B on ALL instances
    - A scores > B on at least one instance
    
    Args:
        candidates: List of candidate indices to check
        pool: CandidatePool with scores
        instance_indices: Instance indices to consider
        
    Returns:
        Set of dominated candidate indices
    """
    dominated = set()
    
    for c1 in candidates:
        if c1 in dominated:
            continue
        
        for c2 in candidates:
            if c1 == c2 or c2 in dominated:
                continue
            
            # Check if c1 dominates c2
            if is_dominated(c2, c1, pool, instance_indices):
                dominated.add(c2)
    
    return dominated


def is_dominated(
    candidate: int,
    by_candidate: int,
    pool: CandidatePool,
    instance_indices: List[int],
) -> bool:
    """
    Check if 'candidate' is dominated by 'by_candidate'.
    
    Args:
        candidate: Candidate that might be dominated
        by_candidate: Candidate that might dominate
        pool: CandidatePool with scores
        instance_indices: Instance indices to consider
        
    Returns:
        True if by_candidate dominates candidate
    """
    all_geq = True  # by_candidate >= candidate on all instances
    any_greater = False  # by_candidate > candidate on at least one
    
    for i in instance_indices:
        score_c = pool.get_score(candidate, i)
        score_by = pool.get_score(by_candidate, i)
        
        # Handle missing scores
        if score_c is None:
            score_c = float('-inf')
        if score_by is None:
            score_by = float('-inf')
        
        if score_by < score_c:
            all_geq = False
            break
        
        if score_by > score_c:
            any_greater = True
    
    return all_geq and any_greater


def compute_pareto_frontier(
    pool: CandidatePool,
    instance_indices: List[int],
) -> List[int]:
    """
    Compute the Pareto frontier (non-dominated candidates).
    
    Args:
        pool: CandidatePool with scores
        instance_indices: Instance indices to consider
        
    Returns:
        List of candidate indices on the Pareto frontier
    """
    all_candidates = list(range(len(pool.candidates)))
    
    if len(all_candidates) == 0:
        return []
    
    dominated = find_dominated_candidates(
        all_candidates,
        pool,
        instance_indices,
    )
    
    return [c for c in all_candidates if c not in dominated]


def compute_hypervolume(
    pool: CandidatePool,
    instance_indices: List[int],
    reference_point: Optional[List[float]] = None,
) -> float:
    """
    Compute the hypervolume indicator for the candidate pool.
    
    This is a measure of the quality of the Pareto frontier.
    Higher hypervolume = better coverage of the objective space.
    
    Args:
        pool: CandidatePool with scores
        instance_indices: Instance indices to consider
        reference_point: Reference point for hypervolume (default: all zeros)
        
    Returns:
        Hypervolume value
    """
    frontier = compute_pareto_frontier(pool, instance_indices)
    
    if not frontier:
        return 0.0
    
    # For simplicity, compute 2D hypervolume using average correctness vs compression
    # In full implementation, this would be multi-dimensional
    
    if reference_point is None:
        reference_point = [0.0, 0.0]
    
    # Collect points (average correctness, average compression) for frontier candidates
    points = []
    for c_idx in frontier:
        scores = [
            pool.get_score(c_idx, i)
            for i in instance_indices
            if pool.get_score(c_idx, i) is not None
        ]
        if scores:
            avg = sum(scores) / len(scores)
            points.append((avg, avg))  # Simplified: using same value for both dims
    
    if not points:
        return 0.0
    
    # Sort points by first coordinate
    points.sort()
    
    # Compute hypervolume (2D)
    hv = 0.0
    prev_y = reference_point[1]
    
    for x, y in points:
        if y > prev_y:
            hv += (x - reference_point[0]) * (y - prev_y)
            prev_y = y
    
    return hv


def get_candidate_statistics(
    pool: CandidatePool,
    instance_indices: List[int],
) -> Dict[str, any]:
    """
    Get statistics about the candidate pool.
    
    Args:
        pool: CandidatePool with scores
        instance_indices: Instance indices to consider
        
    Returns:
        Dictionary with statistics
    """
    if len(pool.candidates) == 0:
        return {
            "num_candidates": 0,
            "num_pareto": 0,
            "best_avg_score": 0.0,
            "worst_avg_score": 0.0,
        }
    
    frontier = compute_pareto_frontier(pool, instance_indices)
    
    # Compute average scores
    avg_scores = []
    for c_idx in range(len(pool.candidates)):
        scores = [
            pool.get_score(c_idx, i)
            for i in instance_indices
            if pool.get_score(c_idx, i) is not None
        ]
        if scores:
            avg_scores.append((c_idx, sum(scores) / len(scores)))
    
    if avg_scores:
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        best_avg = avg_scores[0][1]
        worst_avg = avg_scores[-1][1]
    else:
        best_avg = worst_avg = 0.0
    
    return {
        "num_candidates": len(pool.candidates),
        "num_pareto": len(frontier),
        "pareto_indices": frontier,
        "best_avg_score": best_avg,
        "worst_avg_score": worst_avg,
        "best_candidate_idx": avg_scores[0][0] if avg_scores else 0,
    }
