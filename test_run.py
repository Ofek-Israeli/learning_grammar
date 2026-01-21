#!/usr/bin/env python3
"""Quick test script to run GEPA discovery."""

import sys
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

print("Starting test...")
print(f"Python: {sys.version}")

# Add paths
from pathlib import Path
MINIONS_REPO = Path(__file__).parent.parent / "minions"
sys.path.insert(0, str(MINIONS_REPO))
print(f"Added to path: {MINIONS_REPO}")

try:
    print("Loading config...")
    from kconfig_loader import load_config
    config = load_config('.config')
    print(f"Budget: {config.budget}")
    print(f"Initial logs: {getattr(config, 'initial_logs_dir', 'None')}")
    
    print("\nLoading GEPA components...")
    from gepa_types import BloatAxis, Candidate
    from reflection_lm import ReflectionLM
    from protocol_runner import ProtocolRunner
    
    print("Creating protocol runner...")
    runner = ProtocolRunner(config)
    
    print("\nLoading rollouts from initial logs...")
    initial_logs = getattr(config, 'initial_logs_dir', None)
    if initial_logs:
        dirs = [d.strip() for d in initial_logs.split(',') if d.strip()]
        print(f"Found {len(dirs)} directories")
        
        all_rollouts = []
        for log_dir in dirs:
            logs_path = Path(log_dir)
            if logs_path.is_dir():
                minions_logs = logs_path / "minions_logs"
                if minions_logs.exists():
                    logs_path = minions_logs
                
                loaded = runner.load_from_logs_dir(logs_path, max_logs=None)
                all_rollouts.extend(loaded)
                print(f"  Loaded {len(loaded)} from {logs_path.name}")
        
        print(f"\nTotal rollouts: {len(all_rollouts)}")
        
        # Get transcripts
        transcripts = [r.local_transcript for r in all_rollouts if r.local_transcript]
        print(f"Transcripts with content: {len(transcripts)}")
        
        if transcripts:
            # Use a representative sample - up to 50 transcripts for good coverage
            sample_size = min(50, len(transcripts))
            import random
            random.seed(42)  # For reproducibility
            sampled = random.sample(transcripts, sample_size)
            
            print(f"\nDiscovering axes via GPT-4o using {sample_size} transcripts...")
            reflection = ReflectionLM(model=config.reflection_model)
            axes = reflection.discover_bloat_axes(sampled)
            
            print(f"\n=== Discovered {len(axes)} axes ===")
            for i, axis in enumerate(axes):
                print(f"\n{i+1}. {axis.name}")
                print(f"   Condition: {axis.condition_type.value}")
                print(f"   Penalty: {axis.penalty}")
                print(f"   Phrases: {axis.phrases[:10]}...")
            
            # Tokenize phrases using tiktoken (doesn't require HF auth)
            print("\nTokenizing phrases with tiktoken...")
            try:
                import tiktoken
                # Use cl100k_base encoding (similar to llama tokenizers)
                enc = tiktoken.get_encoding("cl100k_base")
                
                for axis in axes:
                    all_ids = set()
                    for phrase in axis.phrases:
                        # Get token IDs for this phrase
                        tokens = enc.encode(phrase.lower())
                        all_ids.update(tokens)
                        # Also add tokens without leading space
                        tokens_no_space = enc.encode(phrase.lower().strip())
                        all_ids.update(tokens_no_space)
                    axis.token_ids = all_ids
                    print(f"  {axis.name}: {len(axis.phrases)} phrases -> {len(axis.token_ids)} token IDs")
            except ImportError:
                print("  tiktoken not available, using transformers tokenizer if available...")
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained("gpt2")  # Public, no auth needed
                    for axis in axes:
                        all_ids = set()
                        for phrase in axis.phrases:
                            tokens = tok.encode(phrase.lower(), add_special_tokens=False)
                            all_ids.update(tokens)
                        axis.token_ids = all_ids
                        print(f"  {axis.name}: {len(axis.phrases)} phrases -> {len(axis.token_ids)} token IDs")
                except:
                    print("  No tokenizer available - token IDs will be empty!")
            
            # Generate the logit processor
            print("\n" + "="*60)
            print("GENERATING LOGIT PROCESSOR")
            print("="*60)
            
            from output_processor import generate_logit_processor
            from gepa_types import Candidate
            
            # Create a candidate with discovered axes
            candidate = Candidate(
                axes=axes,
                whitelist=set(config.whitelist) if hasattr(config, 'whitelist') and config.whitelist else set(),
                min_new_tokens=48
            )
            
            # Generate to output directory
            output_dir = Path(config.output_dir) if config.output_dir else Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "bloat_axis_logit_processor.py"
            
            generate_logit_processor(candidate, output_file)
            print(f"\nLogit processor written to: {output_file}")
            
            # Also print the content
            print("\n" + "="*60)
            print("LOGIT PROCESSOR CONTENT:")
            print("="*60 + "\n")
            print(output_file.read_text())
            
        else:
            print("No transcripts found!")
    else:
        print("No initial logs configured!")
        
except Exception as e:
    import traceback
    print(f"\nError: {e}")
    traceback.print_exc()

print("\nDone!")
