"""
Reflection LM for Bloat Axis GEPA.

Uses GPT-4o for:
1. Phase 0: Discovering initial bloat axes from verbose transcripts
2. ProposeMutation: Diagnosing under/over-compression and proposing updates
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add minions to path
MINIONS_REPO = Path(__file__).parent.parent / "minions"
sys.path.insert(0, str(MINIONS_REPO))

from minions.clients.openai import OpenAIClient
from minions.usage import Usage

try:
    from .gepa_types import (
        BloatAxis,
        Candidate,
        ConditionType,
        MutationDelta,
        Rollout,
    )
    from .config import (
        PHASE0_DISCOVERY_PROMPT,
        UNDER_COMPRESSION_PROMPT,
        OVER_COMPRESSION_PROMPT,
    )
except ImportError:
    from gepa_types import (
        BloatAxis,
        Candidate,
        ConditionType,
        MutationDelta,
        Rollout,
    )
    from config import (
        PHASE0_DISCOVERY_PROMPT,
        UNDER_COMPRESSION_PROMPT,
        OVER_COMPRESSION_PROMPT,
    )

logger = logging.getLogger(__name__)


class ReflectionLM:
    """
    Reflection language model for bloat axis discovery and mutation proposals.
    
    Uses GPT-4o to analyze transcripts and propose improvements to the
    bloat axis configuration.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the reflection LM.
        
        Args:
            model: Model name (default: gpt-4o)
            api_key: OpenAI API key (uses env var if not provided)
            temperature: Sampling temperature for creative responses
        """
        self.model = model
        self.client = OpenAIClient(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=2000,
        )
        self.total_usage = Usage()
    
    def discover_bloat_axes(
        self,
        transcripts: List[str],
        existing_axes: Optional[List[BloatAxis]] = None,
    ) -> List[BloatAxis]:
        """
        Phase 0: Discover bloat axes from verbose transcripts.
        
        Args:
            transcripts: List of worker transcripts to analyze
            existing_axes: Existing axes to avoid duplicating
            
        Returns:
            List of discovered BloatAxis objects
        """
        # Combine transcripts for analysis
        combined = "\n\n---\n\n".join(transcripts[:5])  # Limit to avoid token overflow
        
        # Build prompt
        prompt = PHASE0_DISCOVERY_PROMPT.format(transcript=combined)
        
        # Get response (client returns List[str], take first)
        responses, usage = self.client.chat([{"role": "user", "content": prompt}])
        self.total_usage += usage
        response = responses[0] if responses else ""
        
        # Parse response
        axes = self._parse_axes_response(response)
        
        # Merge with existing if provided
        if existing_axes:
            axes = self._merge_axes(existing_axes, axes)
        
        logger.info(f"Discovered {len(axes)} bloat axes")
        return axes
    
    def propose_mutation(
        self,
        candidate: Candidate,
        rollouts: List[Rollout],
        axis_idx: Optional[int],
        k_diagnose: int = 3,
        tau: float = 0.7,
    ) -> MutationDelta:
        """
        Propose mutations based on dual-query diagnosis.
        
        Query 1: Top-K longest correct transcripts → under-compression
        Query 2: Top-K shortest incorrect transcripts → over-compression
        
        Args:
            candidate: Current candidate configuration
            rollouts: Rollouts from minibatch
            axis_idx: Index of axis to focus on (None for global)
            k_diagnose: Number of samples for each diagnosis query
            tau: Correctness threshold
            
        Returns:
            MutationDelta with proposed changes
        """
        # Separate correct and incorrect rollouts
        correct_rollouts = [r for r in rollouts if r.is_correct]
        incorrect_rollouts = [r for r in rollouts if not r.is_correct]
        
        # Query 1: Under-compression (longest correct)
        under_delta = self._diagnose_under_compression(
            candidate,
            correct_rollouts,
            k_diagnose,
            axis_idx,
        )
        
        # Query 2: Over-compression (shortest incorrect)
        over_delta = MutationDelta()
        if incorrect_rollouts:
            over_delta = self._diagnose_over_compression(
                candidate,
                incorrect_rollouts,
                k_diagnose,
                axis_idx,
            )
        
        # Merge deltas (over-compression fixes take precedence for safety)
        return under_delta.merge(over_delta)
    
    def _diagnose_under_compression(
        self,
        candidate: Candidate,
        correct_rollouts: List[Rollout],
        k: int,
        axis_idx: Optional[int],
    ) -> MutationDelta:
        """
        Diagnose under-compression from verbose correct answers.
        
        Args:
            candidate: Current candidate
            correct_rollouts: Correctly answered rollouts
            k: Number of longest to analyze
            axis_idx: Axis to focus on (None for all)
            
        Returns:
            MutationDelta for increasing compression
        """
        if not correct_rollouts:
            return MutationDelta()
        
        # Sort by transcript length descending, take top K
        sorted_rollouts = sorted(
            correct_rollouts,
            key=lambda r: r.transcript_length,
            reverse=True
        )[:k]
        
        # Format current axes
        axes_str = self._format_axes(candidate.axes)
        
        # Format verbose responses
        responses_str = "\n\n".join([
            f"Sample: {r.sample_id}\n"
            f"Question: {r.question}\n"
            f"Transcript length: {r.transcript_length} chars\n"
            f"Response: {r.local_transcript[:1000]}..."
            for r in sorted_rollouts
        ])
        
        # Build prompt
        prompt = UNDER_COMPRESSION_PROMPT.format(
            current_axes=axes_str,
            k=k,
            verbose_responses=responses_str,
        )
        
        # Get response (client returns List[str], take first)
        responses, usage = self.client.chat([{"role": "user", "content": prompt}])
        self.total_usage += usage
        response = responses[0] if responses else ""
        
        # Parse into MutationDelta
        return self._parse_under_compression_response(response)
    
    def _diagnose_over_compression(
        self,
        candidate: Candidate,
        incorrect_rollouts: List[Rollout],
        k: int,
        axis_idx: Optional[int],
    ) -> MutationDelta:
        """
        Diagnose over-compression from short incorrect answers.
        
        Args:
            candidate: Current candidate
            incorrect_rollouts: Incorrectly answered rollouts
            k: Number of shortest to analyze
            axis_idx: Axis to focus on (None for all)
            
        Returns:
            MutationDelta for reducing compression
        """
        if not incorrect_rollouts:
            return MutationDelta()
        
        # Sort by transcript length ascending, take top K shortest
        sorted_rollouts = sorted(
            incorrect_rollouts,
            key=lambda r: r.transcript_length,
        )[:k]
        
        # Format current axes
        axes_str = self._format_axes(candidate.axes)
        
        # Format short incorrect responses
        responses_str = "\n\n".join([
            f"Sample: {r.sample_id}\n"
            f"Question: {r.question}\n"
            f"Ground truth: {r.ground_truth[0] if r.ground_truth else 'N/A'}\n"
            f"Transcript length: {r.transcript_length} chars\n"
            f"Response: {r.local_transcript[:1000]}..."
            for r in sorted_rollouts
        ])
        
        # Build prompt
        prompt = OVER_COMPRESSION_PROMPT.format(
            current_axes=axes_str,
            k=k,
            short_responses=responses_str,
        )
        
        # Get response (client returns List[str], take first)
        responses, usage = self.client.chat([{"role": "user", "content": prompt}])
        self.total_usage += usage
        response = responses[0] if responses else ""
        
        # Parse into MutationDelta
        return self._parse_over_compression_response(response)
    
    def _format_axes(self, axes: List[BloatAxis]) -> str:
        """Format axes list for prompt."""
        if not axes:
            return "No axes defined yet."
        
        lines = []
        for i, axis in enumerate(axes):
            cond = f"{axis.condition_type.value}"
            if axis.condition_params:
                cond += f" ({axis.condition_params})"
            
            lines.append(
                f"{i+1}. {axis.name} (penalty={axis.penalty:.1f}, condition={cond})\n"
                f"   Phrases: {', '.join(axis.phrases[:5])}{'...' if len(axis.phrases) > 5 else ''}"
            )
        
        return "\n".join(lines)
    
    def _parse_axes_response(self, response: str) -> List[BloatAxis]:
        """Parse axes discovery response into BloatAxis objects."""
        axes = []
        
        try:
            # Extract JSON from response
            json_obj = self._extract_json(response)
            
            for axis_data in json_obj.get("axes", []):
                condition = axis_data.get("condition", {"type": "always"})
                condition_type = ConditionType(condition.get("type", "always"))
                condition_params = {k: v for k, v in condition.items() if k != "type"}
                
                axis = BloatAxis(
                    name=axis_data.get("name", "unnamed"),
                    phrases=axis_data.get("phrases", []),
                    condition_type=condition_type,
                    condition_params=condition_params,
                    penalty=axis_data.get("penalty", 2.0),
                )
                axes.append(axis)
                
        except Exception as e:
            logger.warning(f"Failed to parse axes response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
        
        return axes
    
    def _parse_under_compression_response(self, response: str) -> MutationDelta:
        """Parse under-compression diagnosis into MutationDelta."""
        delta = MutationDelta()
        
        try:
            json_obj = self._extract_json(response)
            
            # Penalty adjustments (increase penalties)
            for axis_name, penalty in json_obj.get("penalty_adjustments", {}).items():
                delta.penalty_adjustments[axis_name] = float(penalty)
            
            # New axes to add
            for axis_data in json_obj.get("new_axes", []):
                condition = axis_data.get("condition", {"type": "always"})
                condition_type = ConditionType(condition.get("type", "always"))
                condition_params = {k: v for k, v in condition.items() if k != "type"}
                
                axis = BloatAxis(
                    name=axis_data.get("name", "new_axis"),
                    phrases=axis_data.get("phrases", []),
                    condition_type=condition_type,
                    condition_params=condition_params,
                    penalty=axis_data.get("penalty", 2.0),
                )
                delta.new_axes.append(axis)
                
        except Exception as e:
            logger.warning(f"Failed to parse under-compression response: {e}")
        
        return delta
    
    def _parse_over_compression_response(self, response: str) -> MutationDelta:
        """Parse over-compression diagnosis into MutationDelta."""
        delta = MutationDelta()
        
        try:
            json_obj = self._extract_json(response)
            
            # Penalty adjustments (decrease penalties)
            for axis_name, penalty in json_obj.get("penalty_adjustments", {}).items():
                delta.penalty_adjustments[axis_name] = float(penalty)
            
            # Axes to remove
            delta.remove_axes = json_obj.get("remove_axes", [])
            
            # Whitelist additions (phrases to protect)
            # Note: These are phrases, not token IDs yet - will be converted later
            for phrase in json_obj.get("whitelist_additions", []):
                # Store as negative hash temporarily - will be converted to token IDs
                delta.whitelist_additions.add(hash(phrase))
                
        except Exception as e:
            logger.warning(f"Failed to parse over-compression response: {e}")
        
        return delta
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from response text."""
        # Try to find JSON in the response
        response = response.strip()
        
        # Remove markdown code blocks
        if "```" in response:
            # Find JSON block
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Try to extract just the JSON part
                lines = response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{'):
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if line.strip().endswith('}') and in_json:
                        break
                response = '\n'.join(json_lines)
        
        # Parse JSON
        return json.loads(response)
    
    def _merge_axes(
        self,
        existing: List[BloatAxis],
        new: List[BloatAxis],
    ) -> List[BloatAxis]:
        """
        Merge new axes with existing, avoiding duplicates.
        
        Args:
            existing: Existing axes
            new: Newly discovered axes
            
        Returns:
            Merged list of axes
        """
        existing_names = {a.name for a in existing}
        merged = list(existing)
        
        for axis in new:
            if axis.name not in existing_names:
                merged.append(axis)
            else:
                # Merge phrases into existing axis
                for i, ea in enumerate(merged):
                    if ea.name == axis.name:
                        ea.phrases = list(set(ea.phrases) | set(axis.phrases))
                        break
        
        return merged


def apply_mutation(
    candidate: Candidate,
    delta: MutationDelta,
    tokenizer=None,
) -> Candidate:
    """
    Apply a mutation delta to create a new candidate.
    
    Args:
        candidate: Original candidate
        delta: Mutation to apply
        tokenizer: Optional tokenizer for converting phrases to token IDs
        
    Returns:
        New Candidate with mutations applied
    """
    # Create copy
    new_candidate = candidate.copy()
    
    # Apply penalty adjustments
    for axis_name, new_penalty in delta.penalty_adjustments.items():
        for axis in new_candidate.axes:
            if axis.name == axis_name:
                axis.penalty = new_penalty
                break
    
    # Remove axes
    new_candidate.axes = [
        a for a in new_candidate.axes
        if a.name not in delta.remove_axes
    ]
    
    # Add new axes
    for new_axis in delta.new_axes:
        # Compute token IDs if tokenizer provided
        if tokenizer and not new_axis.token_ids:
            new_axis.token_ids = _compute_token_ids(new_axis.phrases, tokenizer)
        new_candidate.axes.append(new_axis)
    
    # Update whitelist
    new_candidate.whitelist |= delta.whitelist_additions
    new_candidate.whitelist -= delta.whitelist_removals
    
    # Update min tokens
    if delta.min_tokens_adjustment is not None:
        new_candidate.min_new_tokens = delta.min_tokens_adjustment
    
    return new_candidate


def _compute_token_ids(phrases: List[str], tokenizer) -> Set[int]:
    """Compute token IDs for a list of phrases."""
    token_ids = set()
    
    for phrase in phrases:
        try:
            # Encode with and without leading space
            ids1 = tokenizer.encode(phrase, add_special_tokens=False)
            ids2 = tokenizer.encode(" " + phrase, add_special_tokens=False)
            
            for tid in ids1 + ids2:
                token_ids.add(int(tid))
        except Exception:
            pass
    
    return token_ids


def select_axis_round_robin(
    candidate: Candidate,
    iteration: int,
) -> Optional[int]:
    """
    Round-robin axis selection for focused mutation.
    
    Args:
        candidate: Current candidate
        iteration: Current iteration number
        
    Returns:
        Axis index to focus on, or None for global mutation
    """
    n_axes = len(candidate.axes)
    if n_axes == 0:
        return None
    
    # Every (n_axes + 1) iterations, do global
    cycle_length = n_axes + 1
    pos = iteration % cycle_length
    
    if pos == n_axes:
        return None  # Global mutation
    
    return pos
