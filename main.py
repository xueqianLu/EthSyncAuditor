"""EthAuditor — main entry point.

Usage:
    python main.py                          # Run with default provider (requires API key)
    python main.py --mock                   # Run with mock agents (no LLM calls)
    python main.py --provider gemini        # Use Gemini (requires GOOGLE_API_KEY)
    python main.py --provider anthropic     # Use Anthropic (requires ANTHROPIC_API_KEY)
    python main.py --resume                 # Resume from latest checkpoint
    python main.py --resume-from 1:5        # Resume from Phase 1, Iteration 5
    python main.py --list-checkpoints       # Show all saved checkpoints
    python main.py --max-iter 3             # Limit both phases to 3 iterations
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from dotenv import load_dotenv

# Load .env file so users can define all environment variables in one place.
# Existing env vars take precedence (override=False is the default).
load_dotenv()

from config import (
    ANTHROPIC_BASE_URL,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OUTPUT_PATH,
)
from file_io.checkpoint import (
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from file_io.writer import (
    write_all_final_lsgs,
    write_diff_report,
    write_diff_report_json,
    write_enriched_spec,
)
from graph import compile_graph, configure_graph, make_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _init_llm(model_name: str, callbacks: list[Any] | None = None,
              provider: str = "anthropic", base_url: str = "") -> Any:
    """Attempt to initialize an LLM for the given *provider*.

    Supported providers: ``"anthropic"`` and ``"gemini"``.

    Parameters
    ----------
    model_name:
        Model identifier to pass to the LLM constructor.
    callbacks:
        Optional LangChain callback handlers.
    provider:
        ``"anthropic"`` or ``"gemini"``.
    base_url:
        Custom API base URL / proxy endpoint.  When non-empty the request
        is routed through this URL instead of the provider's default.
        For Anthropic this maps to ``anthropic_api_url``; for Gemini it
        maps to ``client_options={"api_endpoint": ...}``.

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

        kwargs: dict[str, Any] = {"model": model_name}
        effective_url = base_url or os.environ.get("GOOGLE_API_BASE", "")
        if effective_url:
            kwargs["base_url"] = effective_url
            logger.info("Using custom Gemini API endpoint: %s", effective_url)

        # Forward system proxy settings to httpx via client_args so that the
        # CONNECT tunnel is established correctly (avoids TLS decode errors
        # when http_proxy / https_proxy env vars point to a local proxy).
        proxy_url = (
            os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("all_proxy")
            or os.environ.get("ALL_PROXY")
        )
        if proxy_url:
            kwargs["client_args"] = {"proxy": proxy_url}
            logger.info("Using proxy for Gemini requests: %s", proxy_url)

        llm = ChatGoogleGenerativeAI(**kwargs)
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

    kwargs_a: dict[str, Any] = {"model": model_name}
    effective_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
    if effective_url:
        kwargs_a["anthropic_api_url"] = effective_url
        logger.info("Using custom Anthropic API endpoint: %s", effective_url)

    llm = ChatAnthropic(**kwargs_a)
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
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="PHASE:ITER",
        help="Resume from a specific checkpoint, e.g. --resume-from 1:5",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override MAX_ITER for both phases (e.g. --max-iter 2 for quick test)",
    )
    parser.add_argument(
        "--max-iter-phase1",
        type=int,
        default=None,
        help="Override MAX_ITER_PHASE1 only",
    )
    parser.add_argument(
        "--max-iter-phase2",
        type=int,
        default=None,
        help="Override MAX_ITER_PHASE2 only",
    )
    parser.add_argument(
        "--anthropic-base-url",
        default=None,
        help="Custom API base URL for Anthropic (proxy support)",
    )
    parser.add_argument(
        "--gemini-base-url",
        default=None,
        help="Custom API base URL for Gemini (proxy support)",
    )
    args = parser.parse_args()

    # ── List checkpoints and exit ─────────────────────────────────────
    if args.list_checkpoints:
        ckpts = list_checkpoints()
        if not ckpts:
            print("No checkpoints found.")
        else:
            print(f"{'Phase':<8}{'Iter':<8}{'File'}")
            print("-" * 60)
            for phase, iteration, path in ckpts:
                print(f"{phase:<8}{iteration:<8}{path.name}")
        sys.exit(0)

    provider = args.provider or LLM_PROVIDER

    # ── Apply max-iter overrides ──────────────────────────────────────
    import config as _cfg

    if args.max_iter is not None:
        _cfg.MAX_ITER_PHASE1 = args.max_iter
        _cfg.MAX_ITER_PHASE2 = args.max_iter
    if args.max_iter_phase1 is not None:
        _cfg.MAX_ITER_PHASE1 = args.max_iter_phase1
    if args.max_iter_phase2 is not None:
        _cfg.MAX_ITER_PHASE2 = args.max_iter_phase2

    logger.info("=" * 60)
    logger.info("EthAuditor starting (mock=%s, provider=%s, resume=%s, "
                "max_iter_p1=%d, max_iter_p2=%d)",
                args.mock, provider, args.resume,
                _cfg.MAX_ITER_PHASE1, _cfg.MAX_ITER_PHASE2)
    logger.info("=" * 60)

    # Ensure output dirs exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # ── Audit logger (attached to LLM calls) ──────────────────────────
    # The preprocessor's embedding calls get a "phase0/global" callback.
    # Phase 1/2 agent nodes create their own per-agent callbacks via
    # ``_make_callbacks()`` in graph.py — with the correct phase,
    # iteration, and agent_type metadata.
    callbacks: list[Any] = []
    if not args.mock:
        from file_io.audit_logger import AuditLogCallback

        audit_cb = AuditLogCallback(phase=0, iteration=0, agent_type="preprocess")
        callbacks.append(audit_cb)

    # ── LLM initialization ────────────────────────────────────────────
    llm = None
    use_mock = args.mock
    if not use_mock:
        model_name = GEMINI_MODEL if provider == "gemini" else LLM_MODEL
        base_url = ""
        if provider == "gemini":
            base_url = args.gemini_base_url or GEMINI_BASE_URL
        else:
            base_url = args.anthropic_base_url or ANTHROPIC_BASE_URL
        llm = _init_llm(model_name, callbacks=callbacks, provider=provider,
                         base_url=base_url)
        if llm is None:
            logger.warning("Could not initialize LLM — switching to mock mode")
            use_mock = True

    # ── Configure the graph ───────────────────────────────────────────
    configure_graph(llm=llm, mock=use_mock, callbacks=callbacks)

    # Build and compile graph
    app = compile_graph()

    # ── Initial or resumed state ──────────────────────────────────────
    if args.resume_from:
        # Parse "PHASE:ITER" format
        try:
            parts = args.resume_from.split(":")
            r_phase, r_iter = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            logger.error("Invalid --resume-from format. Use PHASE:ITER, e.g. 1:5")
            sys.exit(1)
        try:
            initial = load_checkpoint(r_phase, r_iter)
            logger.info("Resuming from checkpoint: phase=%d iter=%d", r_phase, r_iter)
        except FileNotFoundError as e:
            logger.error("Checkpoint not found: %s", e)
            logger.info("Use --list-checkpoints to see available checkpoints")
            sys.exit(1)
    elif args.resume:
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
        write_diff_report_json(final_state)

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
