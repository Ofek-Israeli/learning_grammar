"""
Correctness evaluation wrapper for Bloat Axis GEPA.

Checks if correctness is already computed in log files, otherwise invokes
the evaluate/correctness.py script.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add minions to path
MINIONS_REPO = Path(__file__).parent.parent / "minions"
sys.path.insert(0, str(MINIONS_REPO))

from minions.clients.openai import OpenAIClient
from minions.usage import Usage

# Import from our local types module
# Use try/except for flexible import (relative or absolute)
try:
    from .gepa_types import EvaluationResult
except ImportError:
    from gepa_types import EvaluationResult

logger = logging.getLogger(__name__)


# Verdict prompt (same as evaluate/correctness.py)
VERDICT_PROMPT = """You are evaluating whether a predicted answer to a financial question is correct.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

PREDICTED ANSWER:
{predicted}

EVALUATION RULES:
1. For numerical answers: Allow {tolerance:.0%} tolerance (e.g., if ground truth is 100, accept {low}-{high})
2. For yes/no questions: The stance must match exactly
3. For qualitative answers: The key facts and conclusions must align
4. Ignore minor formatting differences (e.g., "$1.5B" vs "1.5 billion dollars")
5. If the predicted answer contains the correct information along with additional context, it's still correct

Respond with a JSON object:
{{
    "is_correct": <true or false>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation of your verdict>"
}}"""


class CorrectnessEvaluator:
    """
    Evaluates correctness of predicted answers.
    
    First checks if correctness is already in the log file.
    If not, uses GPT-4o to evaluate.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        tolerance: float = 0.10,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the correctness evaluator.
        
        Args:
            model: Model to use for evaluation
            tolerance: Numerical tolerance (default 10%)
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.tolerance = tolerance
        self.client = OpenAIClient(
            model_name=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=500,
        )
        self.total_usage = Usage()
    
    def check_log_for_correctness(
        self,
        log_path: Path
    ) -> Optional[EvaluationResult]:
        """
        Check if correctness is already computed in a log file.
        
        Args:
            log_path: Path to the JSON log file
            
        Returns:
            EvaluationResult if found, None otherwise
        """
        if not log_path.exists():
            return None
        
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            # Check for is_correct field
            if 'is_correct' in data and data['is_correct'] is not None:
                return EvaluationResult(
                    is_correct=data['is_correct'],
                    confidence=data.get('correctness_confidence', 1.0),
                    reasoning=data.get('correctness_reasoning', 'From log file')
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        return None
    
    def evaluate(
        self,
        predicted: str,
        ground_truth: List[str],
        question: str,
    ) -> EvaluationResult:
        """
        Evaluate if predicted answer is correct.
        
        Args:
            predicted: The predicted answer
            ground_truth: List of acceptable ground truth answers
            question: The original question
            
        Returns:
            EvaluationResult with verdict
        """
        # Format ground truth
        gt_str = ground_truth[0] if len(ground_truth) == 1 else "\n".join(
            f"- {gt}" for gt in ground_truth
        )
        
        # Compute tolerance bounds for example
        low = int(100 * (1 - self.tolerance))
        high = int(100 * (1 + self.tolerance))
        
        prompt = VERDICT_PROMPT.format(
            question=question,
            ground_truth=gt_str,
            predicted=predicted,
            tolerance=self.tolerance,
            low=low,
            high=high,
        )
        
        try:
            # Client returns List[str], take first
            responses, usage = self.client.chat([{"role": "user", "content": prompt}])
            self.total_usage += usage
            response = responses[0] if responses else ""
            
            # Parse JSON response
            # Try to extract JSON from response
            response_text = response.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(
                    line for line in lines 
                    if not line.startswith('```')
                )
            
            data = json.loads(response_text)
            
            return EvaluationResult(
                is_correct=data.get('is_correct', False),
                confidence=data.get('confidence', 0.5),
                reasoning=data.get('reasoning', 'No reasoning provided'),
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse correctness response: {e}")
            return EvaluationResult(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Failed to parse response: {response[:200]}...",
            )
        except Exception as e:
            logger.error(f"Correctness evaluation failed: {e}")
            return EvaluationResult(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Evaluation error: {str(e)}",
            )
    
    def evaluate_from_log(
        self,
        log_path: Path,
        force_recompute: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate correctness for a sample from its log file.
        
        Args:
            log_path: Path to the minions log JSON file
            force_recompute: If True, recompute even if already in log
            
        Returns:
            EvaluationResult
        """
        # Check if already computed
        if not force_recompute:
            existing = self.check_log_for_correctness(log_path)
            if existing:
                return existing
        
        # Load log file
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        # Extract fields
        question = data.get('task', '')
        predicted = self._extract_predicted_answer(data)
        ground_truth = data.get('ground_truth', [])
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        if not predicted or not ground_truth:
            return EvaluationResult(
                is_correct=False,
                confidence=0.0,
                reasoning="Missing predicted or ground truth answer",
            )
        
        # Evaluate
        result = self.evaluate(predicted, ground_truth, question)
        
        # Update log file with result
        self._update_log_with_result(log_path, result)
        
        return result
    
    def _extract_predicted_answer(self, log_data: Dict[str, Any]) -> str:
        """Extract predicted answer from log data."""
        # Check direct field
        if 'predicted_answer' in log_data:
            return log_data['predicted_answer']
        
        # Look in conversation for final synthesis
        conversation = log_data.get('conversation', [])
        for turn in reversed(conversation):
            if turn.get('user') == 'supervisor':
                output = turn.get('output', '')
                # Try to parse JSON for final_answer field
                if isinstance(output, str):
                    try:
                        parsed = json.loads(output)
                        if 'final_answer' in parsed:
                            return parsed['final_answer']
                    except json.JSONDecodeError:
                        pass
                    # Last supervisor output is often the answer
                    return output
        
        return ""
    
    def _update_log_with_result(
        self,
        log_path: Path,
        result: EvaluationResult
    ):
        """Update log file with correctness result."""
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            data['is_correct'] = result.is_correct
            data['correctness_confidence'] = result.confidence
            data['correctness_reasoning'] = result.reasoning
            
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update log file {log_path}: {e}")


def evaluate_results_dir(
    results_dir: str,
    model: str = "gpt-4o",
    tolerance: float = 0.10,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """
    Evaluate all samples in a results directory.
    
    Args:
        results_dir: Path to results directory
        model: Model to use for evaluation
        tolerance: Numerical tolerance
        verbose: Print detailed results
        
    Returns:
        Tuple of (correct_count, total_count, accuracy)
    """
    results_path = Path(results_dir)
    
    # Check for financebench_results.json
    results_file = results_path / "financebench_results.json"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return 0, 0, 0.0
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    evaluator = CorrectnessEvaluator(model=model, tolerance=tolerance)
    
    correct = 0
    total = 0
    
    for protocol_name, samples in data.get('results', {}).items():
        for sample in samples:
            predicted = sample.get('predicted_answer', '')
            ground_truth = sample.get('ground_truth', [])
            question = sample.get('question', '')
            
            if not predicted or not ground_truth:
                continue
            
            result = evaluator.evaluate(predicted, ground_truth, question)
            total += 1
            
            if result.is_correct:
                correct += 1
            
            if verbose:
                status = "✓" if result.is_correct else "✗"
                print(f"{status} [{result.confidence:.2f}] {sample.get('sample_id', 'unknown')}")
                print(f"   Q: {question[:80]}...")
                print(f"   Predicted: {predicted[:80]}...")
                print(f"   Reasoning: {result.reasoning}")
                print()
    
    accuracy = correct / total if total > 0 else 0.0
    
    if verbose:
        print(f"\nTotal: {correct}/{total} = {accuracy:.1%}")
    
    return correct, total, accuracy


def run_correctness_script(
    results_dir: str,
    model: str = "gpt-4o",
    tolerance: float = 0.10,
    update_summary: bool = True,
    verbose: bool = False,
) -> int:
    """
    Run the evaluate/correctness.py script via subprocess.
    
    Args:
        results_dir: Path to results directory (or run ID)
        model: Remote model to use
        tolerance: Numerical tolerance
        update_summary: Whether to update summary.txt
        verbose: Print detailed results
        
    Returns:
        Return code from subprocess
    """
    script_path = MINIONS_REPO / "evaluate" / "correctness.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        results_dir,
        "--remote-model", model,
        "--tolerance", str(tolerance),
    ]
    
    if update_summary:
        cmd.append("--update-summary")
    
    if verbose:
        cmd.append("--verbose")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Correctness script failed: {result.stderr}")
    elif verbose:
        print(result.stdout)
    
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate correctness")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--model", default="gpt-4o", help="Model for evaluation")
    parser.add_argument("--tolerance", type=float, default=0.10, help="Numerical tolerance")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    correct, total, accuracy = evaluate_results_dir(
        args.results_dir,
        model=args.model,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )
    
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
