# Bloat Axis GEPA

A GEPA-style (Genetic-Pareto) evolutionary optimizer for learning bloat-axis penalties in SGLang logit processors.

## Overview

This module learns to reduce verbosity in LLM worker responses while maintaining correctness. It discovers and optimizes "bloat axes" - patterns of verbose text (preambles, hedging, repetition, etc.) that can be penalized during generation.

## Features

- **Axis Discovery**: Uses GPT-4o to analyze transcripts and discover bloat patterns
- **Pareto Optimization**: Balances compression vs. correctness using Pareto-optimal selection
- **Dual-Query Mutation**: Separate diagnosis for under-compression and over-compression
- **SGLang Integration**: Generates `CustomLogitProcessor` compatible with SGLang servers
- **Kconfig Support**: Flexible configuration via menuconfig TUI

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Run the menuconfig to configure:

```bash
python kmenuconfig.py
```

Or copy `defconfig` to `.config` and edit manually.

## Usage

### Discovery Only (0 iterations)

```bash
python bloat_axis_gepa.py --config .config
```

### Full Evolution

Set `CONFIG_GEPA_BUDGET` to a positive value in `.config`, then run:

```bash
python bloat_axis_gepa.py --config .config
```

## Output

The optimizer generates:
- `output/bloat_axis_logit_processor.py` - SGLang-compatible logit processor
- `output/axes_config.json` - Serialized axes configuration

## Dependencies

- OpenAI API key (for GPT-4o reflection)
- Minions framework (parent repo)
- Optional: SGLang server for live protocol execution

## License

MIT
