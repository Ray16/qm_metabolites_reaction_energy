"""End-to-end smoke test of the QM pipeline (metag.pipeline.score_reaction) on a handful of DIVERSE
reactions (hydratase, kinase/anion, phosphatase, synthase) that exercise different routing paths
(pH-0, truncation, thermal, anchor). Requires the `uma` runtime (torch + fairchem) + a GPU + an xtb
binary, so it is SKIPPED otherwise. Run on a GPU node:  python tests/test_pipeline_smoke.py
"""
import os
import json

_REACTIONS = os.path.join(os.path.dirname(__file__), "data", "smoke_reactions.json")


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_score_several_reactions():
    if not _gpu_available():
        print("SKIP: no GPU / uma runtime")
        return
    from metag.backend.uma import load_uma
    from metag.pipeline import score_reaction
    pu = load_uma(os.environ.get("METAG_MODEL", "uma-s-1p2p1"))
    reactions = json.load(open(_REACTIONS))
    n_ok = 0
    for rid, rx in reactions.items():
        rx = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
        r = score_reaction(pu, rx, seeds=(1,), keep=4, log=lambda *_: None, key=rid)
        # every scored reaction must return a finite dG and a well-formed calibrated interval
        assert r["dG"] is not None and abs(r["dG"]) < 1e4, f"{rid}: bad dG {r['dG']}"
        assert "dG_raw" in r and "sigma_pred" in r and "ci95" in r
        lo, hi = r["ci95"]
        assert lo is None or (lo <= r["dG"] <= hi), f"{rid}: dG outside ci95"
        exp = rx.get("exp")
        err = (r["dG"] - exp[0]) if exp else None
        print(f"  {rid} {rx['note'][7:34]:27s} dG={r['dG']:+7.1f} raw={r['dG_raw']:+7.1f} "
              f"±{r['sigma_pred']}  exp={exp[0] if exp else '?':>6}  err={err if err is None else round(err,1)}")
        n_ok += 1
    assert n_ok == len(reactions)
    print(f"OK: {n_ok}/{len(reactions)} reactions scored end-to-end")


if __name__ == "__main__":
    test_score_several_reactions()
