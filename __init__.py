"""
Bloat Axis GEPA - Evolutionary optimization of logit processors for concise worker messages.

This package implements a GEPA-style (Genetic-Pareto) evolutionary optimizer that discovers
and learns bloat-axis penalties for a local logit processor in a remote-local protocol.

Usage:
    from learning_grammar import BloatAxisGEPA, GEPAConfig
    
    # Configure
    config = GEPAConfig(budget=100, minibatch_size=3)
    
    # Run evolution
    gepa = BloatAxisGEPA(config)
    best_candidate = gepa.evolve(initial_logs_dir="path/to/logs")
    
    # Output is saved to config.output_dir/learned_logit_processor.py
"""

from .gepa_types import (
    BloatAxis,
    Candidate,
    CandidatePool,
    ConditionType,
    ConversationTurn,
    EvaluationResult,
    MutationDelta,
    Rollout,
    ScoreResult,
    WorkerOutput,
)

from .config import GEPAConfig

# Import main class (lazy to avoid circular imports)
def __getattr__(name):
    if name == "BloatAxisGEPA":
        from .bloat_axis_gepa import BloatAxisGEPA
        return BloatAxisGEPA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Main class
    "BloatAxisGEPA",
    # Types
    "BloatAxis",
    "Candidate",
    "CandidatePool",
    "ConditionType",
    "ConversationTurn",
    "EvaluationResult",
    "MutationDelta",
    "Rollout",
    "ScoreResult",
    "WorkerOutput",
    # Config
    "GEPAConfig",
]
