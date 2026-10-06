"""Risk caps, the candidate risk-share cap, constraint sweeps, downside lenses, denoising."""

import copy
import math

import numpy as np
import pytest
import yaml
from riskfolio.src import RiskFunctions as RF

from tests.fixtures.synthetic import fit_context, policy, small_market
from tests.test_runner import SMALL
from workbench.allocators.naive import StaticSAA
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_mr import SkfolioMeanRisk
from workbench.data.synthetic import placeholder_saa
from workbench.evaluation.report import build_report
from workbench.evaluation.views import sweep_view
from workbench.grid.expand import expand
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, parse_spec
from workbench.policy.compiled import candidate_variance_share
from workbench.registry.store import Registry

DATA = small_market()
R = DATA.returns
SAA_W = placeholder_saa()[R.columns]


def _uncapped(rm, obj="MaxRet", cov="hist"):
    return RiskfolioMeanRisk("hist", cov, rm, obj).fit(R, fit_context(DATA)).weights


# --- compile ---------------------------------------------------------------------------------


def test_caps_compile_to_period_units():
    p = policy(max_vol_annual=0.12, max_cvar_period=0.03, max_cdar=0.2, min_return_annual=0.024)
    assert p.max_vol == pytest.approx(0.12 / math.sqrt(12))
    assert p.max_cvar == 0.03 and p.max_cdar == 0.2  # already per period / frequency-free
    assert p.min_return == pytest.approx(0.002)
    assert p.candidate == "CAND"
    with pytest.raises(ValueError, match="max_cdar"):
        policy(max_cdar=0)
    with pytest.raises(ValueError, match="candidate_max_risk_share"):
        policy(candidate_max_risk_share=1.5)


# --- caps bind and the post-check uses the same definitions ---------------------------------


@pytest.mark.parametrize(
    "key,measure",
    [
        ("max_cvar_period", lambda x, w: RF.CVaR_Hist(x, 0.05)),
        ("max_cdar", lambda x, w: RF.CDaR_Abs(x, 0.05)),
        ("max_vol_annual", lambda x, w: math.sqrt(w @ R.cov().to_numpy() @ w) * math.sqrt(12)),
    ],
)
@pytest.mark.parametrize("cls", [RiskfolioMeanRisk, SkfolioMeanRisk])
def test_risk_caps_bind_in_both_libraries(key, measure, cls):
    w0 = _uncapped("MV").to_numpy()
    cap = 0.5 * measure(R.to_numpy() @ w0, w0)
    pol = policy(**{key: cap})
    res = cls("hist", "hist", "MV", "MaxRet").fit(R, fit_context(DATA, pol))
    assert res.status == "ok", res.message
    w = res.weights.to_numpy()
    assert measure(R.to_numpy() @ w, w) <= cap * (1 + 1e-4)
    assert measure(R.to_numpy() @ w, w) >= cap * 0.98  # it binds


def test_post_check_catches_caps_for_heuristics():
    pol = policy(max_cdar=0.01, max_cvar_period=0.001)
    res = StaticSAA().fit(R, fit_context(DATA, pol))
    assert res.status == "infeasible"
    assert "CDaR95" in res.message and "CVaR95" in res.message
    hc = RiskfolioHC().fit(R, fit_context(DATA, policy(min_return_annual=0.5)))
    assert hc.status == "infeasible" and "below floor" in hc.message


# --- candidate risk-share cap ---------------------------------------------------------------


def _attractive_mu():
    mu = R.mean()
    mu["CAND"] = 0.02
    return mu


def test_risk_share_cap_with_mv_and_sample_covariance():
    pol = policy(candidate_max_risk_share=0.05)
    ctx = fit_context(DATA, pol, mu_override=_attractive_mu())
    res = RiskfolioMeanRisk("hist", "hist", "MV", "Sharpe").fit(R, ctx)
    assert res.status == "ok", res.message
    share = candidate_variance_share(res.weights, R.cov().to_numpy(), list(R.columns), "CAND")
    assert share == pytest.approx(0.05, abs=1e-4)


def test_risk_share_cap_never_passed_to_non_mv_and_post_check_holds_it():
    """Riskfolio's arcinequality is meaningless with rm != MV (pushed the candidate to 70%)."""
    pol = policy(candidate_max_risk_share=0.05)
    ctx = fit_context(DATA, pol, mu_override=_attractive_mu())
    res = RiskfolioMeanRisk("hist", "hist", "CVaR", "Sharpe").fit(R, ctx)
    if res.status == "ok":
        share = candidate_variance_share(res.weights, R.cov().to_numpy(), list(R.columns), "CAND")
        assert share <= 0.05 + 1e-6
    else:
        assert res.status == "infeasible" and "variance share" in res.message


def test_risk_share_cap_with_shrunk_covariance_may_breach_sample_definition():
    """Decision 2a: the policy uses the sample covariance; Ledoit-Wolf enforces its own."""
    pol = policy(candidate_max_risk_share=0.05)
    ctx = fit_context(DATA, pol, mu_override=_attractive_mu())
    res = RiskfolioMeanRisk("hist", "ledoit", "MV", "Sharpe").fit(R, ctx)
    assert res.status in ("ok", "infeasible")
    if res.status == "infeasible":
        assert "variance share" in res.message


# --- sweeps ----------------------------------------------------------------------------------


def _sweep_spec(**cs):
    raw = yaml.safe_load(SMALL)
    raw["grid"]["constraint_sets"] = [{"name": "te", **cs}]
    return raw


def test_sweep_expansion_names_and_hash_equivalence():
    swept = parse_spec(_sweep_spec(te_annual={"sweep": [0.01, 0.02]}, candidate_cap=0.1))
    names = [cs.name for cs in swept.constraint_sets]
    assert names == ["te[te_annual=0.01]", "te[te_annual=0.02]"]
    assert swept.constraint_sets[0].sweep == {"base": "te", "keys": {"te_annual": 0.01}}
    raw = yaml.safe_load(SMALL)
    raw["grid"]["constraint_sets"] = [
        {"name": "te[te_annual=0.01]", "te_annual": 0.01, "candidate_cap": 0.1},
        {"name": "te[te_annual=0.02]", "te_annual": 0.02, "candidate_cap": 0.1},
    ]
    explicit = parse_spec(raw)
    assert explicit.spec_hash == swept.spec_hash
    assert [c.config_id for c in expand(explicit)] == [c.config_id for c in expand(swept)]
    assert parse_spec(swept.yaml_text).spec_hash == swept.spec_hash  # dump round-trips


def test_sweep_cartesian_and_errors():
    s = parse_spec(_sweep_spec(te_annual={"sweep": [0.01, 0.02]}, band={"sweep": [0.03, 0.05]}))
    assert len(s.constraint_sets) == 4
    assert s.constraint_sets[1].name == "te[te_annual=0.01,band=0.05]"
    with pytest.raises(SpecError, match="non-empty list"):
        parse_spec(_sweep_spec(te_annual={"sweep": []}))
    with pytest.raises(SpecError, match="duplicate names"):
        raw = _sweep_spec(te_annual={"sweep": [0.01, 0.01]})
        parse_spec(raw)


def test_existing_specs_keep_hash_and_config_ids():
    base = parse_spec(SMALL)
    raw = copy.deepcopy(yaml.safe_load(SMALL))
    raw["grid"]["constraint_sets"][0]["max_cdar"] = 0.3
    assert parse_spec(raw).spec_hash != base.spec_hash
    assert "max_cdar" not in str(base.canonical())


@pytest.fixture(scope="module")
def sweep_run(tmp_path_factory):
    raw = _sweep_spec(te_annual={"sweep": [0.005, 0.02]}, candidate_cap=0.10)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "riskfolio_mean_risk", "rm": ["MV", "CVaR"], "obj": ["Sharpe"]},
    ]
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('sw') / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_sweep_view_and_report(sweep_run):
    reg, s = sweep_run
    v = sweep_view(reg, s.experiment_id)
    full = v[v.data_variant == "full"]
    assert list(full.value) == ["0.005", "0.02"] and set(full.swept) == {"te_annual"}
    assert (full.n_configs_oos >= 1).all()
    assert (full.oos_te_median > 0).all()  # only configurations that traded
    assert (full.oos_te_median.diff().dropna() >= 0).all()  # looser TE budget, more active risk
    md = build_report(reg, s.experiment_id).summary_md
    assert "## Constraint sweeps" in md


# --- downside lenses and denoising -----------------------------------------------------------


@pytest.mark.parametrize("rm", ["MSV", "FLPM", "SLPM"])
@pytest.mark.parametrize("obj", ["MinRisk", "Sharpe"])
def test_downside_measures_match_across_libraries(rm, obj):
    pol = policy(candidate_cap=0.2)
    a = RiskfolioMeanRisk("hist", "hist", rm, obj).fit(R, fit_context(DATA, pol))
    b = SkfolioMeanRisk("hist", "hist", rm, obj).fit(R, fit_context(DATA, pol))
    assert a.status == b.status == "ok"
    assert (a.weights - b.weights).abs().max() < 1e-4


@pytest.mark.parametrize("cov", ["fixed", "spectral", "shrink"])
def test_denoising_estimators(cov):
    res = RiskfolioMeanRisk("hist", cov, "MV", "MinRisk").fit(R, fit_context(DATA))
    assert res.status == "ok", res.message
    assert RiskfolioHC(method_cov=cov).fit(R, fit_context(DATA)).status == "ok"
    if cov == "fixed":
        b = SkfolioMeanRisk("hist", cov, "MV", "MinRisk").fit(R, fit_context(DATA))
        assert (res.weights - b.weights).abs().max() < 1e-3
    else:
        sk = SkfolioMeanRisk("hist", cov).fit(R, fit_context(DATA))
        assert sk.status == "exception" and "unsupported in the skfolio backend" in sk.message


def test_downside_risk_lens_in_runner(tmp_path):
    raw = yaml.safe_load(SMALL)
    raw["risk_lenses"] = ["MV", "MSV", "FLPM"]
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(raw), reg)
    m = reg.metrics(s.experiment_id)
    assert {"MSV", "FLPM"} <= set(m.loc[m.metric == "candidate_risk_share", "lens"])
    assert np.isfinite(m.value).all()


def test_vol_cap_with_risk_share_cap_avoids_riskfolio_bug():
    """Riskfolio 7.4.0: upperdev + arcinequality (MV) raises UnboundLocalError 'g'."""
    pol = policy(max_vol_annual=0.09, candidate_max_risk_share=0.10)
    res = RiskfolioMeanRisk("hist", "ledoit", "MV", "Sharpe").fit(R, fit_context(DATA, pol))
    assert res.status in ("ok", "infeasible"), res.message
    assert "post-check only" in res.diagnostics["vol_cap"]
    if res.status == "ok":
        assert pol.violations(res.weights, R) == []
