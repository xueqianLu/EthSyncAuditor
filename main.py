"""EthAuditor — main entry point.

Usage:
    python main.py          # Run full pipeline (preprocess → Phase 1 → Phase 2)
    python main.py --mock   # Run with mock agents (no LLM calls)
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import OUTPUT_PATH
from graph import compile_graph, make_initial_state
from file_io.checkpoint import save_checkpoint
from file_io.writer import (
    write_all_final_lsgs,
    write_diff_report,
    write_enriched_spec,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="EthAuditor — LSG extraction & comparison")
    parser.add_argument("--mock", action="store_true", help="Run with mock agents (no LLM)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EthAuditor starting (mock=%s)", args.mock)
    logger.info("=" * 60)

    # Ensure output dirs exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Build and compile graph
    app = compile_graph()

    # Initial state
    initial = make_initial_state()

    # Run the graph
    final_state = None
    for step in app.stream(initial, stream_mode="updates"):
        for node_name, state_update in step.items():
            logger.info("Node '%s' completed", node_name)
            if isinstance(state_update, dict):
                final_state = {**(final_state or {}), **state_update}

    if final_state is None:
        logger.error("Graph produced no output")
        sys.exit(1)

    # Write outputs
    phase = final_state.get("current_phase", 0)
    if phase >= 1:
        write_enriched_spec(final_state)
    if phase >= 2:
        write_all_final_lsgs(final_state)
        write_diff_report(final_state)

    # Save final checkpoint
    save_checkpoint(
        final_state,
        phase=final_state.get("current_phase", 0),
        iteration=max(
            final_state.get("phase1_iteration", 0),
            final_state.get("phase2_iteration", 0),
        ),
    )

    force_stopped = final_state.get("force_stopped", False)
    if force_stopped:
        logger.warning("Pipeline finished with FORCE_STOPPED=True")
    else:
        logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
