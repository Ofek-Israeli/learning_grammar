"""
Bloat Axis GEPA - Main Evolution Loop

Implements the GEPA-style evolutionary optimizer for discovering and learning
bloat-axis penalties in a local logit processor.

Based on Algorithm 1 (EvolveBloatAxes) from the algorithm specification.
"""

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add minions to path
MINIONS_REPO = Path(__file__).parent.parent / "minions"
sys.path.insert(0, str(MINIONS_REPO))

try:
    from .gepa_types import (
        BloatAxis,
        Candidate,
        CandidatePool,
        ConditionType,
        MutationDelta,
        Rollout,
        ScoreResult,
    )
    from .config import (
        GEPAConfig,
        DEFAULT_INTRO_PHRASES,
        DEFAULT_ALWAYS_PHRASES,
        DEFAULT_LIST_MARKERS,
        DEFAULT_WHITELIST_PHRASES,
    )
    from .correctness import CorrectnessEvaluator
    from .reflection_lm import ReflectionLM, apply_mutation, select_axis_round_robin
    from .pareto_selection import select_candidate, get_candidate_statistics
    from .protocol_runner import ProtocolRunner, FinanceBenchSample, run_minibatch
    from .output_processor import (
        generate_logit_processor,
        export_axes_config,
        generate_sglang_integration_example,
    )
except ImportError:
    from gepa_types import (
        BloatAxis,
        Candidate,
        CandidatePool,
        ConditionType,
        MutationDelta,
        Rollout,
        ScoreResult,
    )
    from config import (
        GEPAConfig,
        DEFAULT_INTRO_PHRASES,
        DEFAULT_ALWAYS_PHRASES,
        DEFAULT_LIST_MARKERS,
        DEFAULT_WHITELIST_PHRASES,
    )
    from correctness import CorrectnessEvaluator
    from reflection_lm import ReflectionLM, apply_mutation, select_axis_round_robin
    from pareto_selection import select_candidate, get_candidate_statistics
    from protocol_runner import ProtocolRunner, FinanceBenchSample, run_minibatch
    from output_processor import (
        generate_logit_processor,
        export_axes_config,
        generate_sglang_integration_example,
    )

logger = logging.getLogger(__name__)


class BloatAxisGEPA:
    """
    GEPA-style evolutionary optimizer for bloat-axis penalties.
    
    This implements the full evolution loop:
    1. Phase 0: Discover initial bloat axes from verbose transcripts
    2. Initialize candidate pool with discovered axes
    3. Main loop:
       - Pareto-select candidate
       - Gather rollouts on minibatch
       - Dual-query reflection for mutation proposals
       - Test improvement and add to pool if better
    4. Export best candidate as SGLang LogitProcessor
    """
    
    def __init__(
        self,
        config: Optional[GEPAConfig] = None,
    ):
        """
        Initialize the GEPA optimizer.
        
        Args:
            config: Configuration (uses defaults if not provided)
        """
        self.config = config or GEPAConfig()
        
        # Initialize components
        self.protocol_runner = ProtocolRunner(self.config)
        self.reflection_lm = ReflectionLM(model=self.config.reflection_model)
        self.correctness_evaluator = CorrectnessEvaluator(
            model=self.config.reflection_model,
            tolerance=self.config.correctness_tolerance,
        )
        
        # State
        self.pool = CandidatePool()
        self.samples: List[FinanceBenchSample] = []
        self.d_feedback: List[FinanceBenchSample] = []
        self.d_pareto: List[FinanceBenchSample] = []
        self.pareto_indices: List[int] = []
        
        self.iteration = 0
        self.rollout_count = 0
    
    def evolve(
        self,
        initial_logs_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Candidate:
        """
        Run the full evolution loop.
        
        Args:
            initial_logs_dir: Optional path to existing minions_logs for bootstrap
            max_samples: Maximum samples to load
            
        Returns:
            Best candidate from evolution
        """
        logger.info("=" * 60)
        logger.info("Bloat Axis GEPA Evolution Starting")
        logger.info("=" * 60)
        
        # Phase 0: Discover or load initial axes
        initial_axes = self._phase0_discover_axes(initial_logs_dir)
        
        # Initialize candidate pool
        self._initialize_pool(initial_axes)
        
        # If budget is 0, skip evolution and just export initial discovery
        if self.config.budget <= 0:
            logger.info("\n" + "=" * 60)
            logger.info("Discovery-only mode (budget=0)")
            logger.info(f"Discovered {len(initial_axes)} axes")
            for i, axis in enumerate(initial_axes):
                logger.info(f"  {i+1}. {axis.name}: penalty={axis.penalty:.1f}")
            logger.info("=" * 60)
            
            best_candidate = self.pool.candidates[0]
            self._export_results(best_candidate)
            return best_candidate
        
        # Load samples for evolution
        self._load_and_split_data(max_samples)
        
        # Score initial candidate on D_pareto
        self._score_candidate_on_pareto(0)
        
        # Main evolution loop
        while self.rollout_count < self.config.budget:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration} (rollouts: {self.rollout_count}/{self.config.budget}) ---")
            
            # Select candidate via Pareto sampling
            k = select_candidate(self.pool, self.pareto_indices)
            candidate = self.pool.candidates[k]
            
            # Select axis to focus on (round-robin)
            axis_idx = select_axis_round_robin(candidate, self.iteration)
            
            # Sample minibatch from D_feedback
            minibatch = random.sample(
                self.d_feedback,
                min(self.config.minibatch_size, len(self.d_feedback))
            )
            
            # Gather rollouts
            rollouts = self._gather_rollouts(minibatch, candidate)
            self.rollout_count += len(rollouts)
            
            # Evaluate correctness
            self._evaluate_correctness(rollouts)
            
            # Propose mutation via dual-query reflection
            delta = self.reflection_lm.propose_mutation(
                candidate=candidate,
                rollouts=rollouts,
                axis_idx=axis_idx,
                k_diagnose=self.config.k_diagnose,
                tau=self.config.tau,
            )
            
            if delta.is_empty():
                logger.info("No mutations proposed, continuing...")
                continue
            
            # Apply mutation to create new candidate
            new_candidate = apply_mutation(
                candidate,
                delta,
                tokenizer=self.protocol_runner.tokenizer,
            )
            new_candidate.parent_idx = k
            
            # Compute utility before and after
            sigma = self._compute_avg_utility(candidate, rollouts)
            sigma_new = self._compute_avg_utility_new(new_candidate, minibatch)
            
            logger.info(f"Utility: {sigma:.4f} -> {sigma_new:.4f}")
            
            # Accept if improved
            if sigma_new > sigma:
                new_idx = self.pool.add_candidate(new_candidate, parent_idx=k)
                logger.info(f"Added improved candidate {new_idx} (parent={k})")
                
                # Score on D_pareto
                self._score_candidate_on_pareto(new_idx)
                
                # Log statistics
                stats = get_candidate_statistics(self.pool, self.pareto_indices)
                logger.info(
                    f"Pool: {stats['num_candidates']} candidates, "
                    f"{stats['num_pareto']} on Pareto frontier, "
                    f"best avg: {stats['best_avg_score']:.4f}"
                )
            else:
                logger.info("Mutation did not improve, discarding")
        
        # Get best candidate
        best_idx = self.pool.get_best_candidate(self.pareto_indices)
        best_candidate = self.pool.candidates[best_idx]
        
        logger.info("\n" + "=" * 60)
        logger.info("Evolution Complete")
        logger.info(f"Best candidate: {best_idx}")
        logger.info(f"Axes: {len(best_candidate.axes)}")
        for i, axis in enumerate(best_candidate.axes):
            logger.info(f"  {i+1}. {axis.name}: penalty={axis.penalty:.1f}")
        logger.info("=" * 60)
        
        # Export results
        self._export_results(best_candidate)
        
        return best_candidate
    
    def _load_and_split_data(self, max_samples: Optional[int]):
        """Load FinanceBench samples and split into D_feedback and D_pareto."""
        logger.info("Loading FinanceBench samples...")
        
        self.samples = self.protocol_runner.load_samples(max_samples=max_samples)
        
        if len(self.samples) < self.config.n_pareto + self.config.minibatch_size:
            logger.warning(
                f"Not enough samples ({len(self.samples)}) for configured "
                f"n_pareto ({self.config.n_pareto}) + minibatch ({self.config.minibatch_size})"
            )
            self.config.n_pareto = min(self.config.n_pareto, len(self.samples) // 2)
        
        # Shuffle and split
        random.shuffle(self.samples)
        
        self.d_pareto = self.samples[:self.config.n_pareto]
        self.d_feedback = self.samples[self.config.n_pareto:]
        
        # Pareto indices (0 to n_pareto-1)
        self.pareto_indices = list(range(len(self.d_pareto)))
        
        logger.info(f"Split: D_pareto={len(self.d_pareto)}, D_feedback={len(self.d_feedback)}")
    
    def _phase0_discover_axes(
        self,
        initial_logs_dir: Optional[str],
    ) -> List[BloatAxis]:
        """
        Phase 0: Discover initial bloat axes.
        
        If initial_logs_dir is provided, load rollouts from there.
        Supports multiple directories separated by commas.
        Otherwise, run protocols with no penalties to discover axes.
        """
        logger.info("\n--- Phase 0: Discovering Initial Bloat Axes ---")
        
        rollouts: List[Rollout] = []
        
        if initial_logs_dir:
            # Support multiple directories (comma-separated)
            dirs = [d.strip() for d in initial_logs_dir.split(',') if d.strip()]
            
            for log_dir in dirs:
                logs_path = Path(log_dir)
                if logs_path.is_dir():
                    # Check for minions_logs subdirectory
                    minions_logs = logs_path / "minions_logs"
                    if minions_logs.exists():
                        logs_path = minions_logs
                    
                    loaded = self.protocol_runner.load_from_logs_dir(
                        logs_path,
                        max_logs=None,  # Load all from each directory
                    )
                    rollouts.extend(loaded)
                    logger.info(f"Loaded {len(loaded)} rollouts from {logs_path}")
            
            logger.info(f"Total rollouts loaded: {len(rollouts)}")
        
        if not rollouts:
            # Run protocols with no penalties
            logger.info(f"Running {self.config.budget_discover} discovery rollouts...")
            
            discovery_samples = random.sample(
                self.d_feedback,
                min(self.config.budget_discover, len(self.d_feedback))
            )
            
            for sample in discovery_samples:
                logger.info(f"Discovery: {sample.sample_id}")
                rollout = self.protocol_runner.run_protocol(sample, candidate=None)
                rollouts.append(rollout)
                self.rollout_count += 1
        
        # Extract transcripts
        transcripts = [r.local_transcript for r in rollouts if r.local_transcript]
        
        if not transcripts:
            logger.warning("No transcripts for axis discovery, using defaults")
            return self._get_default_axes()
        
        # Discover axes via reflection
        axes = self.reflection_lm.discover_bloat_axes(transcripts)
        
        if not axes:
            logger.warning("No axes discovered, using defaults")
            return self._get_default_axes()
        
        # Compute token IDs
        for axis in axes:
            if not axis.token_ids and self.protocol_runner.tokenizer:
                token_ids = set()
                for phrase in axis.phrases:
                    try:
                        ids1 = self.protocol_runner.tokenizer.encode(
                            phrase, add_special_tokens=False
                        )
                        ids2 = self.protocol_runner.tokenizer.encode(
                            " " + phrase, add_special_tokens=False
                        )
                        for tid in ids1 + ids2:
                            token_ids.add(int(tid))
                    except Exception:
                        pass
                axis.token_ids = token_ids
        
        logger.info(f"Discovered {len(axes)} bloat axes:")
        for axis in axes:
            logger.info(f"  - {axis.name}: {len(axis.phrases)} phrases, {axis.condition_type.value}")
        
        return axes
    
    def _get_default_axes(self) -> List[BloatAxis]:
        """Get default bloat axes when discovery fails."""
        axes = [
            BloatAxis(
                name="intro_preamble",
                phrases=DEFAULT_INTRO_PHRASES,
                condition_type=ConditionType.EARLY_PHASE,
                condition_params={"max_t": 48},
                penalty=self.config.default_penalty,
            ),
            BloatAxis(
                name="filler_words",
                phrases=DEFAULT_ALWAYS_PHRASES,
                condition_type=ConditionType.ALWAYS,
                penalty=self.config.default_penalty / 2,
            ),
            BloatAxis(
                name="list_markers",
                phrases=DEFAULT_LIST_MARKERS,
                condition_type=ConditionType.AFTER_NEWLINE,
                penalty=self.config.default_penalty,
            ),
        ]
        
        # Compute token IDs
        for axis in axes:
            if self.protocol_runner.tokenizer:
                token_ids = set()
                for phrase in axis.phrases:
                    try:
                        ids1 = self.protocol_runner.tokenizer.encode(
                            phrase, add_special_tokens=False
                        )
                        ids2 = self.protocol_runner.tokenizer.encode(
                            " " + phrase, add_special_tokens=False
                        )
                        for tid in ids1 + ids2:
                            token_ids.add(int(tid))
                    except Exception:
                        pass
                axis.token_ids = token_ids
        
        return axes
    
    def _initialize_pool(self, axes: List[BloatAxis]):
        """Initialize candidate pool with first candidate."""
        logger.info("Initializing candidate pool...")
        
        # Compute whitelist token IDs
        whitelist = set()
        if self.protocol_runner.tokenizer:
            for phrase in DEFAULT_WHITELIST_PHRASES:
                try:
                    ids = self.protocol_runner.tokenizer.encode(
                        phrase, add_special_tokens=False
                    )
                    for tid in ids:
                        whitelist.add(int(tid))
                except Exception:
                    pass
        
        # Create initial candidate
        initial_candidate = Candidate(
            axes=axes,
            whitelist=whitelist,
            min_new_tokens=self.config.min_new_tokens,
        )
        
        self.pool.add_candidate(initial_candidate)
        logger.info(f"Initial candidate: {len(axes)} axes, {len(whitelist)} whitelist tokens")
    
    def _gather_rollouts(
        self,
        samples: List[FinanceBenchSample],
        candidate: Candidate,
    ) -> List[Rollout]:
        """Gather rollouts for a minibatch."""
        rollouts = []
        
        for sample in samples:
            logger.info(f"  Running: {sample.sample_id}")
            rollout = self.protocol_runner.run_protocol(sample, candidate)
            rollouts.append(rollout)
        
        return rollouts
    
    def _evaluate_correctness(self, rollouts: List[Rollout]):
        """Evaluate correctness for rollouts that need it."""
        for rollout in rollouts:
            if rollout.is_correct is None:
                result = self.correctness_evaluator.evaluate(
                    predicted=rollout.predicted_answer,
                    ground_truth=rollout.ground_truth,
                    question=rollout.question,
                )
                rollout.is_correct = result.is_correct
                rollout.correctness_confidence = result.confidence
                rollout.correctness_reasoning = result.reasoning
    
    def _compute_avg_utility(
        self,
        candidate: Candidate,
        rollouts: List[Rollout],
    ) -> float:
        """Compute average utility for existing rollouts."""
        utilities = []
        
        for rollout in rollouts:
            score = ScoreResult.compute(
                is_correct=rollout.is_correct or False,
                transcript_length=rollout.transcript_length,
                lambda_=self.config.lambda_,
                tau=self.config.tau,
            )
            if score.utility != float('-inf'):
                utilities.append(score.utility)
        
        return sum(utilities) / len(utilities) if utilities else float('-inf')
    
    def _compute_avg_utility_new(
        self,
        candidate: Candidate,
        samples: List[FinanceBenchSample],
    ) -> float:
        """Compute average utility by running new rollouts."""
        rollouts = self._gather_rollouts(samples, candidate)
        self.rollout_count += len(rollouts)
        
        self._evaluate_correctness(rollouts)
        
        return self._compute_avg_utility(candidate, rollouts)
    
    def _score_candidate_on_pareto(self, candidate_idx: int):
        """Score a candidate on all D_pareto instances."""
        logger.info(f"Scoring candidate {candidate_idx} on D_pareto...")
        
        candidate = self.pool.candidates[candidate_idx]
        
        for i, sample in enumerate(self.d_pareto):
            rollout = self.protocol_runner.run_protocol(sample, candidate)
            self.rollout_count += 1
            
            # Evaluate correctness
            result = self.correctness_evaluator.evaluate(
                predicted=rollout.predicted_answer,
                ground_truth=rollout.ground_truth,
                question=rollout.question,
            )
            
            # Compute score
            score = ScoreResult.compute(
                is_correct=result.is_correct,
                transcript_length=rollout.transcript_length,
                lambda_=self.config.lambda_,
                tau=self.config.tau,
            )
            
            self.pool.set_score(candidate_idx, i, score.utility)
    
    def _export_results(self, best_candidate: Candidate):
        """Export the best candidate as outputs."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate logit processor Python file
        processor_path = output_dir / f"learned_logit_processor_{timestamp}.py"
        generate_logit_processor(best_candidate, processor_path)
        
        # Also save as the default name
        default_path = output_dir / "learned_logit_processor.py"
        generate_logit_processor(best_candidate, default_path)
        
        # Export axes config JSON
        config_path = output_dir / f"axes_config_{timestamp}.json"
        export_axes_config(best_candidate, config_path)
        
        # Generate integration example
        example_path = output_dir / "example_usage.py"
        generate_sglang_integration_example(example_path)
        
        # Save pool statistics
        stats = get_candidate_statistics(self.pool, self.pareto_indices)
        stats_path = output_dir / f"evolution_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "iterations": self.iteration,
                "total_rollouts": self.rollout_count,
                "num_candidates": stats["num_candidates"],
                "num_pareto": stats["num_pareto"],
                "best_avg_score": stats["best_avg_score"],
                "config": {
                    "budget": self.config.budget,
                    "minibatch_size": self.config.minibatch_size,
                    "n_pareto": self.config.n_pareto,
                    "lambda": self.config.lambda_,
                    "tau": self.config.tau,
                },
            }, f, indent=2)
        
        logger.info(f"Results exported to {output_dir}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bloat Axis GEPA - Evolutionary optimization of logit processors"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to Kconfig .config file (alternative to CLI args)",
    )
    parser.add_argument(
        "--initial-logs",
        type=str,
        help="Path to existing minions logs for bootstrap (optional)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to load from FinanceBench",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="Total rollout budget (overrides .config)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        help="Minibatch size (overrides .config)",
    )
    parser.add_argument(
        "--n-pareto",
        type=int,
        help="Size of D_pareto set (overrides .config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (overrides .config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Create config - from .config file or defaults
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = GEPAConfig.from_kconfig(args.config)
        # Get initial_logs from kconfig if set
        initial_logs = getattr(config, 'initial_logs_dir', None)
    else:
        config = GEPAConfig()
        initial_logs = None
    
    # CLI args override .config values
    if args.budget is not None:
        config.budget = args.budget
    if args.minibatch_size is not None:
        config.minibatch_size = args.minibatch_size
    if args.n_pareto is not None:
        config.n_pareto = args.n_pareto
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initial logs: CLI takes precedence
    if args.initial_logs:
        initial_logs = args.initial_logs
    
    # Run evolution
    gepa = BloatAxisGEPA(config)
    best = gepa.evolve(
        initial_logs_dir=initial_logs,
        max_samples=args.max_samples,
    )
    
    print(f"\nBest candidate has {len(best.axes)} axes:")
    for axis in best.axes:
        print(f"  - {axis.name}: penalty={axis.penalty:.1f}, condition={axis.condition_type.value}")


if __name__ == "__main__":
    main()
