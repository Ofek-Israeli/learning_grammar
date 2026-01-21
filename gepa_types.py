"""
Type definitions for Bloat Axis GEPA.

Data classes representing bloat axes, candidates, rollouts, and evaluation results.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import json


class ConditionType(str, Enum):
    """Types of conditions for when to apply bloat axis penalties."""
    ALWAYS = "always"
    EARLY_PHASE = "early_phase"
    AFTER_NEWLINE = "after_newline"
    READY_TO_STOP = "ready_to_stop"


@dataclass
class BloatAxis:
    """
    A bloat axis representing a source of verbosity.
    
    Attributes:
        name: Human-readable name for the axis (e.g., "preamble", "list_starters")
        phrases: List of phrases that contribute to this bloat type
        token_ids: Pre-computed token IDs from the tokenizer
        condition_type: When to apply the penalty
        condition_params: Parameters for the condition (e.g., {"max_t": 48})
        penalty: The penalty magnitude (δ_i)
    """
    name: str
    phrases: List[str]
    token_ids: Set[int] = field(default_factory=set)
    condition_type: ConditionType = ConditionType.ALWAYS
    condition_params: Dict[str, Any] = field(default_factory=dict)
    penalty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "name": self.name,
            "phrases": self.phrases,
            "token_ids": list(self.token_ids),
            "condition": {
                "type": self.condition_type.value,
                **self.condition_params
            },
            "penalty": self.penalty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BloatAxis":
        """Deserialize from dictionary."""
        condition = data.get("condition", {})
        condition_type = ConditionType(condition.get("type", "always"))
        condition_params = {k: v for k, v in condition.items() if k != "type"}
        
        return cls(
            name=data["name"],
            phrases=data.get("phrases", []),
            token_ids=set(data.get("token_ids", [])),
            condition_type=condition_type,
            condition_params=condition_params,
            penalty=data.get("penalty", 1.0)
        )


@dataclass
class Candidate:
    """
    A candidate logit processor configuration.
    
    Attributes:
        axes: List of bloat axes with their penalties
        whitelist: Token IDs that should never be penalized (protected tokens)
        min_new_tokens: Minimum tokens before allowing EOS (anti-truncation)
        parent_idx: Index of parent candidate in pool (None for initial)
    """
    axes: List[BloatAxis]
    whitelist: Set[int] = field(default_factory=set)
    min_new_tokens: int = 48
    parent_idx: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "axes": [axis.to_dict() for axis in self.axes],
            "whitelist": list(self.whitelist),
            "min_new_tokens": self.min_new_tokens,
            "parent_idx": self.parent_idx
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Deserialize from dictionary."""
        return cls(
            axes=[BloatAxis.from_dict(a) for a in data.get("axes", [])],
            whitelist=set(data.get("whitelist", [])),
            min_new_tokens=data.get("min_new_tokens", 48),
            parent_idx=data.get("parent_idx")
        )
    
    def copy(self) -> "Candidate":
        """Create a deep copy of this candidate."""
        return Candidate(
            axes=[BloatAxis(
                name=a.name,
                phrases=list(a.phrases),
                token_ids=set(a.token_ids),
                condition_type=a.condition_type,
                condition_params=dict(a.condition_params),
                penalty=a.penalty
            ) for a in self.axes],
            whitelist=set(self.whitelist),
            min_new_tokens=self.min_new_tokens,
            parent_idx=self.parent_idx
        )


@dataclass
class WorkerOutput:
    """Output from a single worker response."""
    explanation: str
    citation: Optional[str] = None
    answer: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_str: str) -> "WorkerOutput":
        """Parse from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                explanation=data.get("explanation", ""),
                citation=data.get("citation"),
                answer=data.get("answer")
            )
        except json.JSONDecodeError:
            return cls(explanation=json_str)


@dataclass
class ConversationTurn:
    """A single turn in the minions conversation."""
    user: str  # "supervisor" or "worker"
    prompt: str
    output: Any  # str or List[str] for worker batch outputs


@dataclass
class Rollout:
    """
    A single protocol rollout result.
    
    Attributes:
        sample_id: Identifier for the FinanceBench sample
        question: The question being answered
        ground_truth: The expected answer(s)
        predicted_answer: The final answer from the protocol
        conversation: The full conversation trace
        worker_outputs: Parsed worker outputs
        transcript_length: Total length of worker messages (characters)
        token_count: Total tokens in worker outputs
        is_correct: Whether the answer is correct (None if not evaluated)
        correctness_confidence: Confidence of correctness evaluation
        correctness_reasoning: Reasoning for correctness verdict
    """
    sample_id: str
    question: str
    ground_truth: List[str]
    predicted_answer: str
    conversation: List[ConversationTurn] = field(default_factory=list)
    worker_outputs: List[WorkerOutput] = field(default_factory=list)
    transcript_length: int = 0
    token_count: int = 0
    is_correct: Optional[bool] = None
    correctness_confidence: float = 0.0
    correctness_reasoning: str = ""
    
    @property
    def local_transcript(self) -> str:
        """Get concatenated worker outputs as transcript."""
        parts = []
        for wo in self.worker_outputs:
            if wo.explanation:
                parts.append(wo.explanation)
            if wo.answer:
                parts.append(wo.answer)
        return "\n".join(parts)


@dataclass
class EvaluationResult:
    """Result of correctness evaluation."""
    is_correct: bool
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class ScoreResult:
    """Score for a candidate on a single instance."""
    correctness_score: float  # μ_correct
    compression_score: float  # μ_compress
    utility: float  # λ * μ_correct + (1-λ) * μ_compress
    passed_threshold: bool  # Whether μ_correct >= τ
    
    @staticmethod
    def compute(
        is_correct: bool,
        transcript_length: int,
        max_length: int = 10000,
        lambda_: float = 0.6,
        tau: float = 0.7
    ) -> "ScoreResult":
        """
        Compute score from rollout results.
        
        Args:
            is_correct: Whether the answer was correct
            transcript_length: Length of worker transcript
            max_length: Maximum expected transcript length for normalization
            lambda_: Weight for correctness (vs compression)
            tau: Correctness threshold
        """
        correctness_score = 1.0 if is_correct else 0.0
        # Compression score: higher is more concise
        compression_score = max(0.0, 1.0 - (transcript_length / max_length))
        
        passed = correctness_score >= tau
        
        if not passed:
            utility = float('-inf')
        else:
            utility = lambda_ * correctness_score + (1 - lambda_) * compression_score
        
        return ScoreResult(
            correctness_score=correctness_score,
            compression_score=compression_score,
            utility=utility,
            passed_threshold=passed
        )


@dataclass
class MutationDelta:
    """
    A proposed mutation to a candidate.
    
    Attributes:
        penalty_adjustments: Dict of axis_name -> new penalty value
        new_axes: List of new axes to add
        remove_axes: List of axis names to remove
        whitelist_additions: Token IDs to add to whitelist
        whitelist_removals: Token IDs to remove from whitelist
        min_tokens_adjustment: New value for min_new_tokens (None = no change)
    """
    penalty_adjustments: Dict[str, float] = field(default_factory=dict)
    new_axes: List[BloatAxis] = field(default_factory=list)
    remove_axes: List[str] = field(default_factory=list)
    whitelist_additions: Set[int] = field(default_factory=set)
    whitelist_removals: Set[int] = field(default_factory=set)
    min_tokens_adjustment: Optional[int] = None
    
    def merge(self, other: "MutationDelta") -> "MutationDelta":
        """Merge two mutation deltas, with other taking precedence for conflicts."""
        merged_penalties = {**self.penalty_adjustments, **other.penalty_adjustments}
        merged_new_axes = self.new_axes + other.new_axes
        merged_remove = list(set(self.remove_axes) | set(other.remove_axes))
        merged_wl_add = self.whitelist_additions | other.whitelist_additions
        merged_wl_rem = self.whitelist_removals | other.whitelist_removals
        
        # Other's min_tokens takes precedence if set
        min_tokens = other.min_tokens_adjustment if other.min_tokens_adjustment is not None else self.min_tokens_adjustment
        
        return MutationDelta(
            penalty_adjustments=merged_penalties,
            new_axes=merged_new_axes,
            remove_axes=merged_remove,
            whitelist_additions=merged_wl_add,
            whitelist_removals=merged_wl_rem,
            min_tokens_adjustment=min_tokens
        )
    
    def is_empty(self) -> bool:
        """Check if this delta has no changes."""
        return (
            not self.penalty_adjustments and
            not self.new_axes and
            not self.remove_axes and
            not self.whitelist_additions and
            not self.whitelist_removals and
            self.min_tokens_adjustment is None
        )


@dataclass
class CandidatePool:
    """
    Pool of candidates with score tracking.
    
    Attributes:
        candidates: List of candidates
        parents: Parent indices for each candidate
        scores: Score matrix - scores[candidate_idx][instance_idx]
    """
    candidates: List[Candidate] = field(default_factory=list)
    parents: List[Optional[int]] = field(default_factory=list)
    scores: Dict[int, Dict[int, float]] = field(default_factory=dict)
    
    def add_candidate(self, candidate: Candidate, parent_idx: Optional[int] = None) -> int:
        """Add a candidate to the pool and return its index."""
        idx = len(self.candidates)
        candidate.parent_idx = parent_idx
        self.candidates.append(candidate)
        self.parents.append(parent_idx)
        self.scores[idx] = {}
        return idx
    
    def set_score(self, candidate_idx: int, instance_idx: int, score: float):
        """Set the score for a candidate on an instance."""
        if candidate_idx not in self.scores:
            self.scores[candidate_idx] = {}
        self.scores[candidate_idx][instance_idx] = score
    
    def get_score(self, candidate_idx: int, instance_idx: int) -> Optional[float]:
        """Get the score for a candidate on an instance."""
        return self.scores.get(candidate_idx, {}).get(instance_idx)
    
    def get_best_candidate(self, instance_indices: List[int]) -> int:
        """Get index of candidate with highest average score over instances."""
        best_idx = 0
        best_avg = float('-inf')
        
        for c_idx in range(len(self.candidates)):
            scores = [
                self.scores.get(c_idx, {}).get(i, float('-inf'))
                for i in instance_indices
            ]
            valid_scores = [s for s in scores if s != float('-inf')]
            if valid_scores:
                avg = sum(valid_scores) / len(valid_scores)
                if avg > best_avg:
                    best_avg = avg
                    best_idx = c_idx
        
        return best_idx
