"""EthAuditor — main entry point.

Usage:
    python main.py                      # Run with default provider (requires API key)
    python main.py --mock               # Run with mock agents (no LLM calls)
    python main.py --provider gemini    # Use Gemini (requires GOOGLE_API_KEY)
    python main.py --provider anthropic # Use Anthropic (requires ANTHROPIC_API_KEY)
    python main.py --resume             # Resume from latest checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from config import GEMINI_MODEL, LLM_MODEL, LLM_PROVIDER, OUTPUT_PATH
from file_io.checkpoint import latest_checkpoint, save_checkpoint
from file_io.writer import (
    write_all_final_lsgs,
    write_diff_report,
    write_enriched_spec,
)
from graph import compile_graph, configure_graph, make_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _init_llm(model_name: str, callbacks: list[Any] | None = None,
              provider: str = "anthropic") -> Any:
    """Attempt to initialize an LLM for the given *provider*.

    Supported providers: ``"anthropic"`` and ``"gemini"``.

    Returns ``None`` if the required package or API key is not available,
    so the caller can fall back to mock mode gracefully.
    """
    import os

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("langchain-google-genai not installed — falling back to mock mode")
            return None

        if not os.environ.get("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not set — falling back to mock mode")
            return None

        llm = ChatGoogleGenerativeAI(model=model_name, callbacks=callbacks or [])
        logger.info("Initialized Gemini LLM: %s", model_name)
        return llm

    # Default: Anthropic
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("langchain-anthropic not installed — falling back to mock mode")
        return None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — falling back to mock mode")
        return None

    llm = ChatAnthropic(model=model_name, callbacks=callbacks or [])
    logger.info("Initialized Anthropic LLM: %s", model_name)
    return llm


def main() -> None:
    parser = argparse.ArgumentParser(description="EthAuditor — LSG extraction & comparison")
    parser.add_argument("--mock", action="store_true", help="Run with mock agents (no LLM)")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default=None,
        help="LLM provider (default: config.LLM_PROVIDER)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint",
    )
    args = parser.parse_args()

    provider = args.provider or LLM_PROVIDER

    logger.info("=" * 60)
    logger.info("EthAuditor starting (mock=%s, provider=%s, resume=%s)",
                args.mock, provider, args.resume)
    logger.info("=" * 60)

    # Ensure output dirs exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # ── Audit logger (attached to LLM calls) ──────────────────────────
    callbacks: list[Any] = []
    if not args.mock:
        from file_io.audit_logger import AuditLogCallback

        audit_cb = AuditLogCallback(phase=0, iteration=0, agent_type="global")
        callbacks.append(audit_cb)

    # ── LLM initialization ────────────────────────────────────────────
    llm = None
    use_mock = args.mock
    if not use_mock:
        model_name = GEMINI_MODEL if provider == "gemini" else LLM_MODEL
        llm = _init_llm(model_name, callbacks=callbacks, provider=provider)
        if llm is None:
            logger.warning("Could not initialize LLM — switching to mock mode")
            use_mock = True

    # ── Configure the graph ───────────────────────────────────────────
    configure_graph(llm=llm, mock=use_mock, callbacks=callbacks)

    # Build and compile graph
    app = compile_graph()

    # ── Initial or resumed state ──────────────────────────────────────
    if args.resume:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            phase, iteration, initial = ckpt
            logger.info("Resuming from checkpoint: phase=%d iter=%d", phase, iteration)
        else:
            logger.info("No checkpoint found — starting fresh")
            initial = make_initial_state()
    else:
        initial = make_initial_state()

    # ── Run the graph ─────────────────────────────────────────────────
    logger.info("Starting graph execution …")
    final_state = app.invoke(initial)

    if not final_state:
        logger.error("Graph produced no output")
        sys.exit(1)

    # ── Write outputs ─────────────────────────────────────────────────
    phase = final_state.get("current_phase", 0)
    if phase >= 1:
        write_enriched_spec(final_state)
    if phase >= 2:
        write_all_final_lsgs(final_state)
        write_diff_report(final_state)

    # ── Save final checkpoint ─────────────────────────────────────────
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
