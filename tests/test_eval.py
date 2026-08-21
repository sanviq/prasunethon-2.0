"""
Runs the persona eval as part of the suite.

An eval nobody runs is a number in a slide deck. Wiring it into pytest means a
scheme edit that breaks a persona fails immediately, rather than at 3am when
someone regenerates the numbers for the submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import run as evaluation  # noqa: E402


def test_persona_eval_passes(capsys):
    assert evaluation.main() == 0, capsys.readouterr().out


def test_eval_reports_perfect_precision_and_recall(capsys):
    evaluation.main()
    out = capsys.readouterr().out
    assert "MICRO                1.000   1.000   1.000" in out


def test_no_false_rejections_on_partial_information(capsys):
    """The number that matters most on a first turn."""
    evaluation.main()
    assert "false rejections on missing facts: 0/" in capsys.readouterr().out
