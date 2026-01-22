"""
Protocol runner for Bloat Axis GEPA.

Runs the minions protocol on FinanceBench samples with custom logit processor
configurations. Supports both real-time execution and loading from existing logs.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

# Add minions to path
MINIONS_REPO = Path(__file__).parent.parent / "minions"
sys.path.insert(0, str(MINIONS_REPO))

from minions.usage import Usage

# Import minions evaluator modules directly (avoid name collision with learning_grammar's kconfig_loader)
import importlib.util

_minions_kconfig_spec = importlib.util.spec_from_file_location(
    "minions_kconfig_loader", 
    MINIONS_REPO / "evaluate" / "kconfig_loader.py"
)
_minions_kconfig_module = importlib.util.module_from_spec(_minions_kconfig_spec)
_minions_kconfig_spec.loader.exec_module(_minions_kconfig_module)
MinionsKconfigLoader = _minions_kconfig_module.KconfigLoader
MinionsEvaluatorConfig = _minions_kconfig_module.EvaluatorConfig

_fb_evaluator_spec = importlib.util.spec_from_file_location(
    "financebench_evaluator",
    MINIONS_REPO / "evaluate" / "financebench_evaluator.py"
)
_fb_evaluator_module = importlib.util.module_from_spec(_fb_evaluator_spec)
_fb_evaluator_spec.loader.exec_module(_fb_evaluator_module)
FinanceBenchProtocolRunner = _fb_evaluator_module.ProtocolRunner

try:
    from .gepa_types import (
        BloatAxis,
        Candidate,
        ConversationTurn,
        Rollout,
        WorkerOutput,
    )
    from .config import GEPAConfig
except ImportError:
    from gepa_types import (
        BloatAxis,
        Candidate,
        ConversationTurn,
        Rollout,
        WorkerOutput,
    )
    from config import GEPAConfig

logger = logging.getLogger(__name__)


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Concatenated text from all pages
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


class FinanceBenchSample:
    """A single FinanceBench evaluation sample."""
    
    def __init__(
        self,
        sample_id: str,
        question: str,
        ground_truth: List[str],
        document_text: str,
        doc_name: str,
    ):
        self.sample_id = sample_id
        self.question = question
        self.ground_truth = ground_truth
        self.document_text = document_text
        self.doc_name = doc_name


class ProtocolRunner:
    """
    Runs the minions protocol with bloat axis logit processor.
    
    Supports:
    - Real-time protocol execution
    - Loading from existing minions logs
    - Custom logit processor configurations
    """
    
    def __init__(
        self,
        config: GEPAConfig,
        tokenizer=None,
    ):
        """
        Initialize the protocol runner.
        
        Args:
            config: GEPA configuration
            tokenizer: Optional tokenizer for token ID computation
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Try to load tokenizer if not provided
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.local_model)
                logger.info(f"Loaded tokenizer for {config.local_model}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
        
        self.total_usage = Usage()
        
        # Load minions protocol config
        minions_config_path = MINIONS_REPO / ".config"
        if minions_config_path.exists():
            self.minions_eval_config = MinionsKconfigLoader.load_config(str(minions_config_path))
            logger.info(f"Loaded minions config from {minions_config_path}")
        else:
            self.minions_eval_config = None
            logger.warning("minions/.config not found, using defaults")
        
        # Create the financebench protocol runner for running minions
        mc = self.minions_eval_config
        if mc:
            self.fb_protocol_runner = FinanceBenchProtocolRunner(
                local_model=mc.models.local.name,
                remote_model=mc.models.remote.name,
                local_temp=mc.models.local.temperature,
                remote_temp=mc.models.remote.temperature,
                num_ctx=mc.models.local.num_ctx,
                local_backend=mc.models.local.backend,
                sglang_base_url=mc.models.local.sglang_base_url,
                generation_strategy=mc.models.local.generation_strategy,
                beta_explanation=mc.models.local.beta_explanation,
                beta_citation=mc.models.local.beta_citation,
                beta_answer=mc.models.local.beta_answer,
                min_tokens_explanation=mc.models.local.min_tokens_explanation,
                min_tokens_citation=mc.models.local.min_tokens_citation,
                min_tokens_answer=mc.models.local.min_tokens_answer,
                max_tokens_explanation=mc.models.local.max_tokens_explanation,
                max_tokens_citation=mc.models.local.max_tokens_citation,
                max_tokens_answer=mc.models.local.max_tokens_answer,
            )
        else:
            # Fallback to defaults
            self.fb_protocol_runner = FinanceBenchProtocolRunner(
                local_model=config.local_model,
                remote_model=config.reflection_model,
                local_backend="sglang",
                sglang_base_url=config.sglang_base_url,
            )
    
    def load_samples(
        self,
        max_samples: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
    ) -> List[FinanceBenchSample]:
        """
        Load FinanceBench samples from dataset.
        
        Args:
            max_samples: Maximum number of samples to load
            sample_indices: Specific indices to load (1-based)
            
        Returns:
            List of FinanceBenchSample objects
        """
        dataset_path = Path(self.config.dataset_path)
        pdf_dir = Path(self.config.pdf_dir)
        
        samples = []
        
        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                # Filter by indices if specified
                if sample_indices and line_num not in sample_indices:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Get question
                    question = data.get('question', '')
                    if not question:
                        continue
                    
                    # Get answer
                    answer = data.get('answer', [])
                    if isinstance(answer, str):
                        answer = [answer]
                    
                    # Get doc_name
                    doc_name = data.get('doc_name')
                    if not doc_name:
                        continue
                    
                    # Load PDF text
                    pdf_path = pdf_dir / f"{doc_name}.pdf"
                    if not pdf_path.exists():
                        logger.warning(f"PDF not found: {pdf_path}")
                        continue
                    
                    document_text = load_pdf_text(str(pdf_path))
                    
                    sample = FinanceBenchSample(
                        sample_id=f"financebench_line_{line_num}",
                        question=question,
                        ground_truth=answer,
                        document_text=document_text,
                        doc_name=doc_name,
                    )
                    samples.append(sample)
                    
                    if max_samples and len(samples) >= max_samples:
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
        
        logger.info(f"Loaded {len(samples)} FinanceBench samples")
        return samples
    
    def run_protocol(
        self,
        sample: FinanceBenchSample,
        candidate: Optional[Candidate] = None,
    ) -> Rollout:
        """
        Run the minions protocol on a single sample using financebench_evaluator.
        
        Args:
            sample: FinanceBench sample to run
            candidate: Logit processor configuration (None for no penalties)
            
        Returns:
            Rollout with results and traces
        """
        # Build kwargs from minions config
        minions_proto = self.minions_eval_config.protocols.minions if self.minions_eval_config else None
        common_proto = self.minions_eval_config.protocols.common if self.minions_eval_config else None
        
        kwargs = {
            'max_chunk_size': minions_proto.max_chunk_size if minions_proto else 3000,
            'pages_per_chunk': minions_proto.pages_per_chunk if minions_proto else 5,
            'num_tasks_per_round': minions_proto.num_tasks_per_round if minions_proto else 12,
            'num_samples_per_task': common_proto.num_samples_per_task if common_proto else 1,
            'chunk_fn': minions_proto.chunk_fn if minions_proto else "chunk_on_multiple_pages",
            'max_jobs_per_round': minions_proto.max_jobs_per_round if minions_proto else None,
        }
        
        # Extract custom token IDs from candidate if provided
        if candidate and candidate.axes:
            self._ensure_token_ids(candidate)
            
            intro_token_ids = set()
            always_token_ids = set()
            list_token_ids = set()
            
            for axis in candidate.axes:
                if axis.condition_type.value == "early_phase":
                    intro_token_ids |= axis.token_ids
                elif axis.condition_type.value == "always":
                    always_token_ids |= axis.token_ids
                elif axis.condition_type.value == "after_newline":
                    list_token_ids |= axis.token_ids
            
            if intro_token_ids:
                kwargs['intro_token_ids'] = list(intro_token_ids)
            if always_token_ids:
                kwargs['always_token_ids'] = list(always_token_ids)
            if list_token_ids:
                kwargs['list_token_ids'] = list(list_token_ids)
        
        # Run protocol using financebench_evaluator's ProtocolRunner
        max_rounds = common_proto.max_rounds if common_proto else self.config.max_rounds
        
        try:
            result = self.fb_protocol_runner.run_minions(
                question=sample.question,
                context=sample.document_text,
                max_rounds=max_rounds,
                **kwargs
            )
            
            # Parse result
            predicted_answer = result.get('final_answer', '')
            
            # Update usage
            self.total_usage += result.get('local_usage', Usage())
            self.total_usage += result.get('remote_usage', Usage())
            
            rollout = Rollout(
                sample_id=sample.sample_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                predicted_answer=predicted_answer,
                is_correct=None,  # Will be evaluated later
            )
            
            return rollout
            
        except Exception as e:
            logger.error(f"Protocol failed for {sample.sample_id}: {e}")
            return Rollout(
                sample_id=sample.sample_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                predicted_answer="",
                is_correct=False,
            )
    
    def _ensure_token_ids(self, candidate: Candidate):
        """Ensure all axes have token IDs computed."""
        if self.tokenizer is None:
            return
        
        for axis in candidate.axes:
            if not axis.token_ids:
                token_ids = set()
                for phrase in axis.phrases:
                    try:
                        ids1 = self.tokenizer.encode(phrase, add_special_tokens=False)
                        ids2 = self.tokenizer.encode(" " + phrase, add_special_tokens=False)
                        for tid in ids1 + ids2:
                            token_ids.add(int(tid))
                    except Exception:
                        pass
                axis.token_ids = token_ids
    
    def _extract_answer(self, result: Dict[str, Any]) -> str:
        """Extract final answer from minions result."""
        if 'final_answer' in result:
            return str(result['final_answer'])
        
        # Try to find in final_output
        final_output = result.get('final_output', {})
        if isinstance(final_output, dict):
            if 'answer' in final_output:
                return str(final_output['answer'])
            if 'final_answer' in final_output:
                return str(final_output['final_answer'])
        
        return str(final_output) if final_output else ""
    
    def _extract_conversation(
        self,
        result: Dict[str, Any],
    ) -> List[ConversationTurn]:
        """Extract conversation turns from result."""
        conversation = []
        
        for turn in result.get('conversation', []):
            conversation.append(ConversationTurn(
                user=turn.get('user', 'unknown'),
                prompt=turn.get('prompt', ''),
                output=turn.get('output', ''),
            ))
        
        return conversation
    
    def _extract_worker_outputs(
        self,
        conversation: List[ConversationTurn],
    ) -> List[WorkerOutput]:
        """Extract worker outputs from conversation."""
        outputs = []
        
        for turn in conversation:
            if turn.user == 'worker':
                # Worker outputs can be a list
                if isinstance(turn.output, list):
                    for out_str in turn.output:
                        outputs.append(WorkerOutput.from_json(out_str))
                elif isinstance(turn.output, str):
                    outputs.append(WorkerOutput.from_json(turn.output))
        
        return outputs
    
    def _compute_transcript(self, outputs: List[WorkerOutput]) -> str:
        """Compute transcript string from worker outputs."""
        parts = []
        for wo in outputs:
            if wo.explanation:
                parts.append(wo.explanation)
            if wo.answer:
                parts.append(wo.answer)
        return "\n".join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Rough estimate if no tokenizer
        return len(text) // 4
    
    def load_from_log(
        self,
        log_path: Path,
    ) -> Rollout:
        """
        Load a rollout from an existing minions log file.
        
        Args:
            log_path: Path to the minions log JSON file
            
        Returns:
            Rollout loaded from log
        """
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        # Extract data
        sample_id = log_path.stem
        question = data.get('task', '')
        
        # Extract conversation
        conversation = []
        for turn in data.get('conversation', []):
            conversation.append(ConversationTurn(
                user=turn.get('user', 'unknown'),
                prompt=turn.get('prompt', ''),
                output=turn.get('output', ''),
            ))
        
        # Extract worker outputs
        worker_outputs = self._extract_worker_outputs(conversation)
        transcript = self._compute_transcript(worker_outputs)
        
        # Extract answer
        predicted_answer = data.get('predicted_answer', '')
        if not predicted_answer:
            predicted_answer = self._extract_answer(data)
        
        # Extract ground truth
        ground_truth = data.get('ground_truth', [])
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Check for correctness
        is_correct = data.get('is_correct')
        confidence = data.get('correctness_confidence', 0.0)
        reasoning = data.get('correctness_reasoning', '')
        
        return Rollout(
            sample_id=sample_id,
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            conversation=conversation,
            worker_outputs=worker_outputs,
            transcript_length=len(transcript),
            token_count=self._count_tokens(transcript),
            is_correct=is_correct,
            correctness_confidence=confidence,
            correctness_reasoning=reasoning,
        )
    
    def load_from_logs_dir(
        self,
        logs_dir: Path,
        max_logs: Optional[int] = None,
    ) -> List[Rollout]:
        """
        Load rollouts from a minions_logs directory.
        
        Args:
            logs_dir: Path to minions_logs directory
            max_logs: Maximum number of logs to load
            
        Returns:
            List of Rollouts
        """
        rollouts = []
        
        if not logs_dir.exists():
            logger.warning(f"Logs directory not found: {logs_dir}")
            return rollouts
        
        log_files = sorted(logs_dir.glob("*.json"))
        
        if max_logs:
            log_files = log_files[:max_logs]
        
        for log_path in log_files:
            try:
                rollout = self.load_from_log(log_path)
                rollouts.append(rollout)
            except Exception as e:
                logger.warning(f"Failed to load log {log_path}: {e}")
        
        logger.info(f"Loaded {len(rollouts)} rollouts from {logs_dir}")
        return rollouts


def run_minibatch(
    runner: ProtocolRunner,
    samples: List[FinanceBenchSample],
    candidate: Optional[Candidate],
) -> List[Rollout]:
    """
    Run protocol on a minibatch of samples.
    
    Args:
        runner: ProtocolRunner instance
        samples: Samples to run
        candidate: Candidate configuration
        
    Returns:
        List of Rollouts
    """
    rollouts = []
    
    for sample in samples:
        logger.info(f"Running protocol on {sample.sample_id}...")
        rollout = runner.run_protocol(sample, candidate)
        rollouts.append(rollout)
    
    return rollouts
