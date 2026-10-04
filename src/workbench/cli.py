"""Command line: ``wb run specs/<name>.yaml`` and ``wb report <experiment>``.

The registry URL comes from ``--registry``, else ``$WB_REGISTRY``, else
``sqlite:///out/registry.db``. Reports go to ``<out>/<experiment name>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from workbench.evaluation.report import build_report, headline_text
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, load_spec
from workbench.registry.store import Registry

DEFAULT_REGISTRY = "sqlite:///out/registry.db"
log = logging.getLogger("workbench")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        registry = _open_registry(args.registry)
        if args.command == "run":
            return _run(args, registry)
        return _report(args, registry)
    except (SpecError, KeyError, FileNotFoundError, NotImplementedError) as e:
        sys.stderr.write(f"wb: error: {e}\n")
        return 2


def _run(args, registry: Registry) -> int:
    spec = load_spec(args.spec)
    summary = run_experiment(spec, registry, if_exists=args.if_exists)
    report = build_report(registry, summary.experiment_id)
    paths = report.write(_out_dir(args.out, report.name))
    _emit(
        f"experiment {report.name} ({summary.experiment_id})"
        f"{' [already in registry, not re-run]' if summary.skipped else ''}\n"
        f"cells: {summary.n_cells} {dict(sorted(summary.status_counts.items()))}\n\n"
        f"{headline_text(report.corridor)}\n\n" + "".join(f"wrote {p}\n" for p in paths.values())
    )
    return 0


def _report(args, registry: Registry) -> int:
    experiment_id = registry.resolve(args.experiment)
    report = build_report(registry, experiment_id)
    paths = report.write(_out_dir(args.out, report.name))
    _emit(f"{headline_text(report.corridor)}\n\n" + "".join(f"wrote {p}\n" for p in paths.values()))
    return 0


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", default=os.environ.get("WB_REGISTRY", DEFAULT_REGISTRY),
                        help="SQLAlchemy URL (default: $WB_REGISTRY or %(default)s)")  # fmt: skip
    common.add_argument("--out", default="out", help="output root directory (default: out)")
    common.add_argument("-v", "--verbose", action="store_true", help="log progress")

    p = argparse.ArgumentParser(prog="wb", description="Portfolio construction workbench")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="run an experiment spec")
    run.add_argument("spec", type=Path, help="path to specs/<name>.yaml")
    run.add_argument("--if-exists", choices=["skip", "replace", "error"], default="skip",
                     help="when the same spec + data vintage is already registered")  # fmt: skip
    rep = sub.add_parser("report", parents=[common], help="rebuild a report from the registry")
    rep.add_argument("experiment", help="experiment name (latest run) or experiment_id")
    return p


def _open_registry(url: str) -> Registry:
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return Registry(url)


def _out_dir(root: str, name: str) -> Path:
    return Path(root) / re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _emit(text: str) -> None:
    """User-facing CLI output (the product of the command, not a log message)."""
    sys.stdout.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
