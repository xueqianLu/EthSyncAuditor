# EthSyncAuditor

An automated auditing system that extracts and cross-compares **Logic Synchronization Graphs (LSGs)** from Ethereum consensus-layer client implementations. EthSyncAuditor uses LLM-powered multi-agent collaboration (via [LangGraph](https://github.com/langchain-ai/langgraph)) and RAG (Retrieval-Augmented Generation) to model each client's internal workflows as state machines, then identifies behavioral differences across implementations.

## Supported Clients

| Client | Language |
|---|---|
| [Prysm](https://github.com/prysmaticlabs/prysm) | Go |
| [Lighthouse](https://github.com/sigp/lighthouse) | Rust |
| [Grandine](https://github.com/grandinetech/grandine) | Rust |
| [Teku](https://github.com/Consensys/teku) | Java |
| [Lodestar](https://github.com/chainsafe/lodestar) | TypeScript |

## What is an LSG?

A **Logic Synchronization Graph** is a state-machine model of an Ethereum consensus client's internal business workflows. Each LSG consists of:

- **Guards** — Boolean preconditions that gate state transitions (e.g. `RespRecv`, `TimeoutExpired`, `HasProposerDuty`)
- **Actions** — Side effects executed during transitions (e.g. `SendRangeRequest`, `ValidateBatch`, `SignAttestation`)
- **Workflows** — Directed state machines covering 7 core scenarios:

| Workflow ID | Description |
|---|---|
| `initial_sync` | Bootstrap from empty state to chain head |
| `regular_sync` | Steady-state sync via gossip |
| `checkpoint_sync` | Fast sync from a trusted checkpoint |
| `attestation_generate` | Create and broadcast attestations |
| `block_generate` | Build and propose blocks |
| `aggregate` | Aggregate attestation subnets |
| `execute_layer_relation` | Coordinate with the execution layer |

See [`docs/LSG_Schema_Spec.md`](docs/LSG_Schema_Spec.md) for the full schema specification.

## Architecture

EthSyncAuditor is built as a two-phase LangGraph pipeline with iterative convergence:

```
preprocess → Phase 1 (vocabulary discovery) → Phase 2 (LSG extraction & diff)
```

### Phase 1 — Vocabulary Discovery
Five per-client sub-agents analyze source code to discover Guard/Action vocabulary. A main agent merges discoveries and computes a `diff_rate`. The loop iterates until convergence (`diff_rate < 0.05`) or a maximum iteration limit is reached.

### Phase 2 — LSG Extraction & Comparison
Five per-client sub-agents extract full LSG state machines using the converged vocabulary. A main agent cross-compares all client LSGs, classifies differences (A-class semantic vs. B-class structural), and computes a `logic_diff_rate`. The loop iterates until convergence or the iteration limit.

Each phase fans out work to all 5 clients in parallel using LangGraph's [Send API](https://langchain-ai.github.io/langgraph/concepts/low_level/#send), then aggregates results through a main agent node.

See [`docs/Architecture.md`](docs/Architecture.md) for the full system architecture, state dictionary specification, and node topology diagram.

## Installation

### Prerequisites

- Python 3.12+
- An API key for at least one supported LLM provider

### Setup

```bash
# Clone the repository
git clone https://github.com/xueqianLu/EthSyncAuditor.git
cd EthSyncAuditor

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download client source code for analysis
bash getcode.sh
```

### Environment Variables

EthSyncAuditor supports a **`.env` file** in the project root for convenient
configuration.  Copy the provided template and fill in the values you need:

```bash
cp .env.example .env
# Edit .env with your API keys and optional proxy URLs
```

Alternatively, export the variables in your shell — shell environment variables
always take precedence over `.env` values.

```bash
# For Anthropic (default)
export ANTHROPIC_API_KEY="your-key-here"

# For Google Gemini
export GOOGLE_API_KEY="your-key-here"
```

#### API Proxy / Custom Endpoints

If you need to route LLM requests through a proxy or custom endpoint, set the
appropriate base URL via environment variable or CLI flag:

```bash
# Anthropic proxy (env var or CLI flag)
export ANTHROPIC_BASE_URL="https://your-proxy.example.com/anthropic"
python main.py --anthropic-base-url https://your-proxy.example.com/anthropic

# Gemini proxy (env var or CLI flag)
export GOOGLE_API_BASE="https://your-proxy.example.com/gemini"
python main.py --gemini-base-url https://your-proxy.example.com/gemini
```

CLI flags take precedence over config defaults; environment variables are used
as a final fallback when neither is set.

## Usage

```bash
# Run with default provider (Anthropic)
python main.py

# Run with Gemini
python main.py --provider gemini

# Run in mock mode (no LLM calls — deterministic stubs)
python main.py --mock

# Resume from the latest checkpoint
python main.py --resume

# List all available checkpoints
python main.py --list-checkpoints

# Resume from a specific checkpoint (e.g. Phase 1, Iteration 5)
python main.py --resume-from 1:5

# Limit both phases to N iterations (useful for quick testing)
python main.py --max-iter 3

# Limit iterations per phase independently
python main.py --max-iter-phase1 5 --max-iter-phase2 3

# Combine: resume from a checkpoint with a reduced iteration limit
python main.py --resume-from 1:8 --max-iter 2
```

### CLI Reference

| Flag | Description |
|---|---|
| `--mock` | Run with mock agents (no LLM calls) |
| `--provider {anthropic,gemini}` | LLM provider (default: `config.LLM_PROVIDER`) |
| `--resume` | Resume from the latest checkpoint |
| `--resume-from PHASE:ITER` | Resume from a specific checkpoint, e.g. `1:5` |
| `--list-checkpoints` | List all saved checkpoints and exit |
| `--max-iter N` | Override max iterations for **both** phases |
| `--max-iter-phase1 N` | Override max iterations for Phase 1 only |
| `--max-iter-phase2 N` | Override max iterations for Phase 2 only |
| `--anthropic-base-url URL` | Custom API base URL for Anthropic (proxy support) |
| `--gemini-base-url URL` | Custom API base URL for Gemini (proxy support) |

### Output

Results are written to the `output/` directory:

```
output/
├── preprocess/                         # AST symbols, call graphs, search indexes
├── checkpoints/                        # Per-iteration state snapshots
├── iterations/                         # Intermediate LSG YAML per client per iteration
├── audit_logs/                         # LLM call audit trails
├── Global_LSG_Spec_Enriched.yaml       # Phase 1 output: enriched vocabulary
├── LSG_<client>_final.yaml             # Phase 2 output: per-client LSG
└── Audit_Diff_Report.md                # Phase 2 output: cross-client diff report
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_graph.py -v
```

## Project Structure

```
EthSyncAuditor/
├── main.py                  # CLI entry point
├── config.py                # Centralized configuration constants
├── graph.py                 # LangGraph pipeline definition (nodes, edges, routers)
├── state.py                 # GlobalState TypedDict & Pydantic sub-models
├── utils.py                 # Serialization helpers
├── agents/                  # LLM agent factory functions
│   ├── phase1_sub_agent.py  # Per-client vocabulary discovery
│   ├── phase1_main_agent.py # Vocabulary merge & diff computation
│   ├── phase2_sub_agent.py  # Per-client LSG extraction
│   ├── phase2_main_agent.py # Cross-client LSG comparison
│   └── prompts/             # Jinja2 prompt templates
├── tools/                   # Preprocessing & retrieval tools
│   ├── preprocessor.py      # AST parsing, call graphs, index building
│   └── search.py            # Hybrid BM25 + vector search
├── file_io/                 # File I/O utilities
│   ├── checkpoint.py        # State checkpoint save/load
│   ├── writer.py            # YAML/Markdown output writers
│   └── audit_logger.py      # LangChain callback for LLM audit logs
├── docs/                    # Documentation
│   ├── Architecture.md      # System architecture & node topology
│   └── LSG_Schema_Spec.md   # LSG schema specification
├── tests/                   # Pytest test suite
├── getcode.sh               # Script to clone client repositories
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Pytest configuration
```

## License

See the repository for license information.