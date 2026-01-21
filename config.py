"""
Configuration for Bloat Axis GEPA.

Hyperparameters, paths, and default settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Base paths - relative to minions repo
MINIONS_REPO = Path(__file__).parent.parent / "minions"
EVALUATE_DIR = MINIONS_REPO / "evaluate"
RESULTS_DIR = EVALUATE_DIR / "results"


@dataclass
class GEPAConfig:
    """
    Configuration for the GEPA evolution loop.
    
    Attributes:
        # Data paths
        dataset_path: Path to FinanceBench dataset JSON
        pdf_dir: Directory containing PDF files
        output_dir: Directory for output files
        
        # Budget and sampling
        budget: Total rollout budget
        budget_discover: Budget for Phase 0 axis discovery
        minibatch_size: Size of minibatch for each iteration
        n_pareto: Size of D_pareto (Pareto evaluation set)
        k_diagnose: Number of samples for diagnosis queries
        
        # Optimization weights
        lambda_: Weight for correctness vs compression (higher = more correctness)
        tau: Correctness threshold (minimum acceptable correctness)
        
        # Logit processor defaults
        default_penalty: Initial penalty for discovered axes
        min_new_tokens: Minimum tokens before allowing EOS
        
        # Model settings
        reflection_model: Model for reflection LM (GPT-4o)
        local_model: Model for local worker
        sglang_base_url: URL for SGLang server
        
        # Protocol settings
        max_rounds: Maximum communication rounds
        chunk_size: Chunk size for document segmentation
    """
    # Data paths
    dataset_path: str = str(EVALUATE_DIR / "data" / "financebench_open_source.jsonl")
    pdf_dir: str = str(EVALUATE_DIR / "data" / "financebench_pdfs")
    output_dir: str = str(Path(__file__).parent / "output")
    
    # Budget and sampling
    budget: int = 100
    budget_discover: int = 5
    minibatch_size: int = 3
    n_pareto: int = 20
    k_diagnose: int = 3
    
    # Optimization weights
    lambda_: float = 0.6  # Correctness weight
    tau: float = 0.7  # Correctness threshold
    
    # Logit processor defaults
    default_penalty: float = 2.0
    min_new_tokens: int = 48
    
    # Model settings
    reflection_model: str = "gpt-4o"
    local_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    sglang_base_url: str = "http://localhost:8000"
    
    # Protocol settings
    max_rounds: int = 5
    chunk_size: int = 3000
    
    # Correctness evaluation
    correctness_tolerance: float = 0.10
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "GEPAConfig":
        """Create config from environment variables."""
        return cls(
            dataset_path=os.getenv("GEPA_DATASET_PATH", cls.dataset_path),
            pdf_dir=os.getenv("GEPA_PDF_DIR", cls.pdf_dir),
            output_dir=os.getenv("GEPA_OUTPUT_DIR", cls.output_dir),
            budget=int(os.getenv("GEPA_BUDGET", cls.budget)),
            budget_discover=int(os.getenv("GEPA_BUDGET_DISCOVER", cls.budget_discover)),
            minibatch_size=int(os.getenv("GEPA_MINIBATCH_SIZE", cls.minibatch_size)),
            n_pareto=int(os.getenv("GEPA_N_PARETO", cls.n_pareto)),
            k_diagnose=int(os.getenv("GEPA_K_DIAGNOSE", cls.k_diagnose)),
            lambda_=float(os.getenv("GEPA_LAMBDA", cls.lambda_)),
            tau=float(os.getenv("GEPA_TAU", cls.tau)),
            default_penalty=float(os.getenv("GEPA_DEFAULT_PENALTY", cls.default_penalty)),
            min_new_tokens=int(os.getenv("GEPA_MIN_NEW_TOKENS", cls.min_new_tokens)),
            reflection_model=os.getenv("GEPA_REFLECTION_MODEL", cls.reflection_model),
            local_model=os.getenv("GEPA_LOCAL_MODEL", cls.local_model),
            sglang_base_url=os.getenv("SGLANG_BASE_URL", cls.sglang_base_url),
            max_rounds=int(os.getenv("GEPA_MAX_ROUNDS", cls.max_rounds)),
            chunk_size=int(os.getenv("GEPA_CHUNK_SIZE", cls.chunk_size)),
            correctness_tolerance=float(os.getenv("GEPA_CORRECTNESS_TOLERANCE", cls.correctness_tolerance)),
        )
    
    @classmethod
    def from_kconfig(cls, config_path: str) -> "GEPAConfig":
        """
        Load config from a Kconfig .config file.
        
        Args:
            config_path: Path to .config file
            
        Returns:
            GEPAConfig loaded from the file
        """
        try:
            from .kconfig_loader import KconfigLoader
        except ImportError:
            from kconfig_loader import KconfigLoader
        return KconfigLoader.load_config(config_path)


# Default whitelist phrases - tokens that should never be penalized
# These are typically important for factual content
DEFAULT_WHITELIST_PHRASES = [
    # Numbers and currency
    "$", "%", "million", "billion", "thousand",
    # Common financial terms
    "revenue", "profit", "loss", "margin", "ratio",
    "assets", "liabilities", "equity", "debt",
    # Years
    "2020", "2021", "2022", "2023", "2024",
    "FY2020", "FY2021", "FY2022", "FY2023", "FY2024",
]


# Default intro phrases to penalize (verbose preambles)
DEFAULT_INTRO_PHRASES = [
    # Meta-commentary starters
    "The task requires", "I will focus", "I was able to", "I can provide",
    "Let me", "I'll", "We need to", "We will", "We can",
    "Based on", "In order to", "To find", "To extract", "To determine",
    # Hedging starters
    "Unfortunately", "However", "Nevertheless", "Therefore",
    # "Not found" verbose patterns
    "The required", "The provided", "The document", "The excerpt",
    "is not explicitly", "is not directly", "was not found",
    "does not contain", "does not include", "does not provide",
    # Excessive politeness
    "To address", "Additionally", "It's also important", 
    "Pay attention to", "Look for", "focus on extracting",
    "This could include", "Please provide", "In this implementation",
]


# Default always-penalized phrases (discourse markers/filler)
DEFAULT_ALWAYS_PHRASES = [
    # Filler words
    "basically", "actually", "really", "just", "overall", "generally",
    "in summary", "to summarize", "in conclusion", "of course", "certainly", "sure",
    "I think", "I can", "I will",
    # Hedging/transition words
    "however", "unfortunately", "nevertheless", "therefore", "furthermore",
    "additionally", "moreover", "consequently", "hence",
    # Filler phrases
    "it appears", "it seems", "we can infer", "we can assume",
    "might be", "could be", "may be", "likely", "possibly",
    "in other words", "that is to say", "specifically",
    # Empty qualifiers
    "explicitly", "directly", "specifically mentioned",
]


# Default list markers to penalize
DEFAULT_LIST_MARKERS = [
    "\n1", "\n2", "\n3", "\n4", "\n5",
    "\n- ", "\n* ", "\n• ",
    "\nFirst", "\nSecond", "\nThird", "\nNext", "\nThen",
    "Step 1", "Step 2", "Step 3",
    "1.", "2.", "3.", "4.", "5.",
    "1)", "2)", "3)", "4)", "5)",
]


# Reflection prompts
PHASE0_DISCOVERY_PROMPT = """Analyze the following worker responses from a document QA system. 
The worker receives chunks of financial documents and answers questions about them.

Your task: Identify patterns of verbosity and unnecessary content in the worker's responses.

Worker Transcript:
{transcript}

---

Please identify bloat axes - categories of unnecessary verbiage that could be reduced.
For each axis, provide:
1. A short name (e.g., "preamble", "list_markers", "hedging")
2. Example phrases that belong to this axis
3. When this bloat typically occurs (always, early in response, after newlines, etc.)

Respond with a JSON object:
{{
  "axes": [
    {{
      "name": "axis_name",
      "phrases": ["phrase1", "phrase2", ...],
      "condition": {{"type": "always"}} or {{"type": "early_phase", "max_t": 48}} or {{"type": "after_newline"}}
    }},
    ...
  ]
}}"""


UNDER_COMPRESSION_PROMPT = """You are building a logit processor with bloat axes to make worker responses more concise.

These worker responses are CORRECT but VERBOSE - they contain unnecessary words, filler, preambles, etc.
The responses should be concise but NOT truncated.

Current bloat axes being penalized:
{current_axes}

Verbose responses (top {k} longest):
{verbose_responses}

---

Which bloat axis is UNDER-PENALIZED or MISSING? 
What changes would make these responses more concise without losing correctness?

Respond with a JSON object:
{{
  "diagnosis": "explanation of what verbosity remains",
  "penalty_adjustments": {{"axis_name": new_penalty_value, ...}},
  "new_axes": [
    {{
      "name": "new_axis_name",
      "phrases": ["phrase1", "phrase2", ...],
      "condition": {{"type": "always"}}
    }}
  ]
}}"""


OVER_COMPRESSION_PROMPT = """You are building a logit processor with bloat axes to make worker responses more concise.

These worker responses are INCORRECT and SHORT - they may have been truncated or had important information penalized away.

Current bloat axes being penalized:
{current_axes}

Short incorrect responses (top {k} shortest incorrect):
{short_responses}

---

Which bloat axis caused INFORMATION LOSS?
What changes would fix the over-compression (reduce penalties, add to whitelist)?

Respond with a JSON object:
{{
  "diagnosis": "explanation of what information was lost",
  "penalty_adjustments": {{"axis_name": reduced_penalty_value, ...}},
  "remove_axes": ["axis_name_to_remove", ...],
  "whitelist_additions": ["important_phrase_to_protect", ...]
}}"""
