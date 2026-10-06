"""IC memo: decision block, kill-criterion replay, checks, the memo from the registry, CLI."""

import math

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import yaml

from tests.test_runner import SMALL
from workbench.cli import main
from workbench.evaluation.memo import FLAG, Facts, build_memo, kill_triggers, memo_checks
from workbench.grid.runner import run_experiment
from workbench.grid.spec import KillCriterion, SpecError, parse_spec
from workbench.registry.store import Registry

DECISION = {
    "candidate_name": "Trend X",
    "recommendation": "Allocate 5%, corridor 2-8%.",
    "proposal": {"weight": 0.05},
    "target_corridor": [0.02, 0.08],
    "conditions": ["Rebalance quarterly."],
    "kill_criteria": [
        {"metric": "te_vs_saa", "above": 0.004, "months": 12},
        {"metric": "active_return", "below": -0.01, "months": 24},
        {"text": "Key-person event"},
    ],
    "owner": "CIO office",
    "review": "2027-06",
}


def _raw(decision=DECISION):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "saa_plus", "x": [0.05]},
        {"type": "riskfolio_mean_risk", "rm": ["MV"], "obj": ["Sharpe", "MinRisk"]},
    ]
    raw["stress"] = {"windows": {"mid": ["2017-01", "2017-06"]}, "weights": [0.05],
                     "bootstrap": {"n_paths": 200, "horizon_years": 5}}  # fmt: skip
    if decision is not None:
        raw["decision"] = decision
    return raw


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("memo")
    reg = Registry(f"sqlite:///{path / 'r.db'}")
    return reg, run_experiment(parse_spec(_raw()), reg), path


# --- decision block ---------------------------------------------------------------------------


def test_decision_is_not_hashed_and_round_trips():
    with_d, without = parse_spec(_raw()), parse_spec(_raw(None))
    assert with_d.spec_hash == without.spec_hash
    assert "decision" not in with_d.canonical()
    d = with_d.decision
    assert d.proposal_weight == 0.05 and d.proposal_funding == "pro_rata"
    assert d.kill_criteria[0] == KillCriterion("te_vs_saa", 0.004, None, 12)
    assert d.kill_criteria[2].describe() == "Key-person event"
    assert d.kill_criteria[1].describe() == "active_return < -1.00% over 24 months"
    assert parse_spec(with_d.yaml_text).decision == d


@pytest.mark.parametrize(
    "change,match",
    [
        ({"proposal": {"weight": 1.2}}, "proposal.weight"),
        ({"proposal": {"weight": 0.05, "funding": "cash"}}, "funding"),
        ({"target_corridor": [0.08, 0.02]}, "target_corridor"),
        ({"conditions": "text"}, "conditions"),
        ({"kill_criteria": [{"metric": "vol", "above": 0.1, "months": 12}]}, "metric"),
        ({"kill_criteria": [{"metric": "te_vs_saa", "above": 0.1, "below": 0.0, "months": 12}]},
         "exactly one"),
        ({"kill_criteria": [{"metric": "te_vs_saa", "above": 0.1, "months": 1}]}, "months"),
        ({"vote": "yes"}, "unknown keys"),
    ],
)  # fmt: skip
def test_decision_errors(change, match):
    with pytest.raises(SpecError, match=match):
        parse_spec(_raw({**DECISION, **change}))


# --- kill criteria ----------------------------------------------------------------------------


def test_kill_triggers_match_a_hand_calculation():
    idx = pd.date_range("2015-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(0)
    saa = pd.Series(rng.normal(0.004, 0.02, 40), index=idx)
    port = saa + pd.Series(rng.normal(0.0, 0.002, 40), index=idx)
    port.iloc[30:] -= 0.004  # a run of underperformance
    te = KillCriterion("te_vs_saa", above=0.008, months=12)
    r = kill_triggers(port, saa, te, "M")
    hand = [(port - saa).iloc[i - 12 : i].std(ddof=1) * math.sqrt(12) for i in range(12, 41)]
    assert r["n_windows"] == 29 and r["n_triggered"] == sum(v > 0.008 for v in hand)
    assert r["worst"] == pytest.approx(max(hand))
    ar = KillCriterion("active_return", below=-0.02, months=12)
    r = kill_triggers(port, saa, ar, "M")
    hand = [(1 + port.iloc[i - 12:i]).prod() - (1 + saa.iloc[i - 12:i]).prod()
            for i in range(12, 41)]  # fmt: skip
    trig = [v < -0.02 for v in hand]
    assert r["n_triggered"] == sum(trig) > 0
    assert r["first"] == idx[11 + trig.index(True)].date().isoformat()
    assert "not a whole number" in kill_triggers(port, saa, KillCriterion(
        "te_vs_saa", above=0.01, months=4), "Q")["note"]  # fmt: skip


# --- checks -----------------------------------------------------------------------------------


CORR = {"p10": 0.0, "p25": 0.02, "median": 0.04, "p75": 0.06, "p90": 0.10}


def _flags(**kw) -> dict:
    t = memo_checks(Facts(**{"x_ref": 0.04, "x_source": "proposal", "corridor_full": CORR, **kw}))
    return dict(zip(t["check"], t["flag"], strict=True))


def _flag(flags: dict, start: str) -> str:
    (v,) = [f for c, f in flags.items() if c.startswith(start)]
    return v


@pytest.mark.parametrize(
    "good,bad,start",
    [
        ({"x_ref": 0.04}, {"x_ref": 0.08}, "Proposal within the corridor"),
        ({"target_corridor": (0.02, 0.08)}, {"target_corridor": (0.02, 0.15)}, "Target corridor"),
        ({"n_configs": 10, "n_significant": 1}, {"n_configs": 10, "n_significant": 0},
         "Configurations beating"),
        ({"proposal_p_bh": 0.01}, {"proposal_p_bh": 0.2}, "The proposal's own"),
        ({"dsr_best": 0.97}, {"dsr_best": 0.5}, "Deflated Sharpe"),
        ({"pbo": 0.3}, {"pbo": 0.7}, "Probability of backtest"),
        ({"spanning_p": 0.01}, {"spanning_p": 0.3}, "Spanning"),
        ({"oos_periods": 60}, {"oos_periods": 20}, "Out-of-sample length"),
        ({"p_worse": 0.1, "stress_x": 0.04}, {"p_worse": 0.8, "stress_x": 0.04}, "Stress"),
        ({"required_excess": 0.01, "cma_excess": 0.03},
         {"required_excess": 0.04, "cma_excess": 0.03}, "Expected excess return"),
        ({"agreement_max_diff": 0.001, "agreement_mismatches": 0},
         {"agreement_max_diff": 0.05, "agreement_mismatches": 0}, "Library agreement"),
        ({"failed_share": 0.1}, {"failed_share": 0.9}, "Failed cells"),
        ({"backfilled_share": 0.0}, {"backfilled_share": 0.8}, "Backfilled share"),
    ],
)  # fmt: skip
def test_each_check_in_both_directions(good, bad, start):
    assert _flag(_flags(**good), start) == ""
    assert _flag(_flags(**bad), start) == FLAG


def test_missing_evidence_is_not_evaluated_never_flagged():
    t = memo_checks(Facts(x_ref=0.03, x_source="corridor median"))
    assert (t["flag"] == "").all()
    assert (t["value"] == "not evaluated").sum() >= 8


# --- the memo from the registry ---------------------------------------------------------------


def test_memo_from_the_registry_alone(run, monkeypatch):
    reg, s, _ = run

    def no_data(*a, **k):
        raise AssertionError("the memo must not load market data")

    monkeypatch.setattr("workbench.grid.runner.load_market", no_data)
    memo = build_memo(reg, s.experiment_id)
    md = memo.markdown
    assert memo.status == "PROPOSED"
    for heading in (
        "# IC memo: Trend X in the SAA",
        "## Recommendation",
        "### Checks",
        "## Situation",
        "## Complication",
        "## Question",
        "## Answer",
        "### 1. The corridor",
        "### 2. Evidence net of search",
        "### 3. Risk at the proposal",
        "### 4. Return assumptions",
        "## Conditions",
        "## Kill criteria",
        "## Robustness",
        "## Appendix",
    ):
        assert heading in md, heading  # fmt: skip
    assert "> Allocate 5%, corridor 2-8%." in md and "- Rebalance quarterly." in md
    assert memo.facts.proposal_p_bh is not None and memo.facts.p_worse is not None
    assert memo.facts.stress_x == pytest.approx(0.05)
    assert "Standalone profiles" in md and "| Key-person event | qualitative |" in md


def test_kill_criteria_in_the_memo_replay_the_stored_path(run):
    reg, s, _ = run
    memo = build_memo(reg, s.experiment_id)
    cells = reg.cells(s.experiment_id)
    cid = cells[(cells.allocator == "saa_plus") & (cells.data_variant == "full")].config_id.iloc[0]
    p = reg.oos_returns(s.experiment_id, include_reference=True).query("data_variant == 'full'")
    wide = p.assign(date=pd.to_datetime(p.date)).pivot(
        index="date", columns="config_id", values="portfolio_return_net")  # fmt: skip
    k = parse_spec(_raw()).decision.kill_criteria[0]
    r = kill_triggers(wide[cid], wide["saa_reference"], k, "M")
    assert f"fired in {r['n_triggered']} of {r['n_windows']} windows" in memo.markdown


def test_draft_without_a_decision(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(_raw(None)), reg)
    memo = build_memo(reg, s.experiment_id)
    assert memo.status == "DRAFT" and memo.facts.x_source == "corridor median"
    assert "To be written by the proposer" in memo.markdown
    assert "## Kill criteria\n\n_None stated._" in memo.markdown


def test_revised_spec_supplies_the_decision_and_must_match(run):
    reg, s, path = run
    revised = _raw({**DECISION, "recommendation": "Allocate 4% after review."})
    good = path / "revised.yaml"
    good.write_text(yaml.safe_dump(revised, sort_keys=False))
    memo = build_memo(reg, s.experiment_id, spec_path=good)
    assert "> Allocate 4% after review." in memo.markdown and str(good) in memo.markdown
    changed = _raw()
    changed["grid"]["constraint_sets"][0]["candidate_cap"] = 0.2
    bad = path / "changed.yaml"
    bad.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="spec_hash differs"):
        build_memo(reg, s.experiment_id, spec_path=bad)


def test_old_experiment_without_profiles(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(_raw()), reg)
    with reg.engine.begin() as c:
        c.execute(sa.text("delete from evidence where test = 'profile'"))
    md = build_memo(reg, s.experiment_id).markdown
    assert "profiles not stored for this experiment" in md


def test_cli_writes_the_memo(run, capsys):
    reg, s, path = run
    url = str(reg.engine.url)
    assert main(["memo", s.experiment_id, "--registry", url, "--out", str(path / "out")]) == 0
    out = capsys.readouterr().out
    assert "PROPOSED" in out and "checks flagged" in out
    assert (path / "out" / "runner_test" / "memo.md").exists()
    bad = path / "other.yaml"
    other = _raw()
    other["seed"] = 8
    bad.write_text(yaml.safe_dump(other, sort_keys=False))
    rc = main(["memo", s.experiment_id, "--registry", url, "--spec", str(bad)])
    assert rc == 2 and "spec_hash differs" in capsys.readouterr().err
