"""11_regime_conditional_rules: do any rules pay off ONLY in specific regimes?

Depends on:
  - full_history_canonical/TRADES_<SYM>_full_history.csv
  - research/04_loss_region_models/tree_leaves_<sym>.csv (the candidate rules)

If item 04 is rerun and its output schema changes, parse_path / apply_conditions
here may break — the path string format from item 04 is contractually:
    "feat OP value AND feat OP value AND ..."
where OP is `<=` or `>` and value is a float (no scientific notation).


Take candidate rules discovered in item 04 (tree_leaves with precision_is>=0.65
and n_is>=30). For each rule, evaluate per (regime × research-OOS year). Then
construct meta-rules of the form:

    "apply rule R only when regime == G"

Score these on the RESEARCH_OOS window (2021-2025) and report the EMBARGO
result (2026) WITHOUT iterating against it.

This is high overfit risk: each (rule, regime) pair gets thin n. We gate with:
  - minimum trigger n >= 8 across RESEARCH_OOS
  - random-baseline comparison at the same trigger rate
  - report EMBARGO numbers ONCE per rule, never as a selection criterion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _rule_naming import rule_hash, rule_name  # noqa: E402


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
ITEM_04 = ROOT.parent / "04_loss_region_models"
OUT = ROOT
SEED = 42
IS_END_YEAR = 2020
RESEARCH_OOS_END = 2025

REGIMES = ("bull", "chop_highvol", "sideways_lowvol")


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    df["split"] = np.where(df["year"] <= IS_END_YEAR, "IS",
                  np.where(df["year"] <= RESEARCH_OOS_END, "RESEARCH_OOS", "EMBARGO"))
    return df


def parse_path(path_str: str) -> list[tuple[str, str, float]]:
    if not isinstance(path_str, str) or not path_str.strip():
        return []
    conds = []
    for part in path_str.split(" AND "):
        # Bounded splits: defensive against any future feature name containing
        # the operator substring. Today none of our features do.
        if " <= " in part:
            f, v = part.split(" <= ", 1)
            conds.append((f.strip(), "<=", float(v)))
        elif " > " in part:
            f, v = part.split(" > ", 1)
            conds.append((f.strip(), ">", float(v)))
    return conds


def apply_conditions(df: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for f, op, thr in conditions:
        if f.startswith("regime_"):
            regime = f.replace("regime_", "")
            v = (df["regime_entry"] == regime).astype(int)
        else:
            v = df.get(f)
            if v is None:
                return pd.Series(False, index=df.index)
        m = (v <= thr) if op == "<=" else (v > thr)
        mask &= m.fillna(False)
    return mask


def random_baseline_precision(df: pd.DataFrame, trigger_rate: float, n_iters: int, rng: np.random.Generator) -> tuple[float, float]:
    if trigger_rate <= 0 or df.empty:
        return np.nan, np.nan
    losers = df["is_loser"].values
    pnls = df["pnl_pct"].values
    precs = []
    nets = []
    for _ in range(n_iters):
        m = rng.random(len(df)) < trigger_rate
        if m.sum() == 0:
            continue
        precs.append(float(losers[m].mean()))
        nets.append(float(-pnls[m].sum()))
    return (float(np.median(precs)) if precs else np.nan,
            float(np.median(nets)) if nets else np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cell_rows = []
    meta_rule_rows = []

    for sym in ("TQQQ", "SQQQ"):
        leaves_path = ITEM_04 / f"tree_leaves_{sym}.csv"
        if not leaves_path.exists():
            print(f"WARN: {leaves_path} missing — run item 04 first.")
            continue
        leaves = pd.read_csv(leaves_path)
        candidates = leaves[leaves["is_candidate_rule"] == True].copy()
        if candidates.empty:
            continue
        df = load(sym)

        for _, rule in candidates.iterrows():
            conds = parse_path(rule["path"])
            if not conds:
                continue
            rname = rule.get("rule_name") if "rule_name" in rule and pd.notna(rule.get("rule_name")) else f"{sym}_{rule['tree']}_{rule['leaf_id']}"

            for split in ("RESEARCH_OOS", "EMBARGO"):
                sub = df[df["split"] == split]
                if sub.empty:
                    continue
                for regime in REGIMES:
                    rsub = sub[sub["regime_entry"] == regime]
                    if len(rsub) < 5:
                        continue
                    mask = apply_conditions(rsub, conds)
                    n_f = int(mask.sum())
                    flagged = rsub[mask]
                    base_loser = float(rsub["is_loser"].mean())
                    trig = float(mask.mean())
                    rand_prec, rand_net = random_baseline_precision(rsub, trig, 300, rng)
                    cell_rows.append({
                        "symbol": sym,
                        "rule_name": rname,
                        "tree": rule.get("tree"),
                        "target": rule.get("target"),
                        "split": split,
                        "regime": regime,
                        "n_regime_total": len(rsub),
                        "n_flagged": n_f,
                        "trigger_rate": trig,
                        "precision_loser": float(flagged["is_loser"].mean()) if n_f else np.nan,
                        "net_pnl_pct_impact": float(-flagged["pnl_pct"].sum()) if n_f else 0.0,
                        "mean_flagged_pnl_pct": float(flagged["pnl_pct"].mean()) if n_f else np.nan,
                        "random_baseline_precision": rand_prec,
                        "random_baseline_net_impact": rand_net,
                        "precision_minus_random": (float(flagged["is_loser"].mean()) - rand_prec) if n_f and not np.isnan(rand_prec) else np.nan,
                    })

            # Construct meta-rules: rule R conditional on regime G
            for regime in REGIMES:
                meta_name = f"{rname}__given_{regime}"
                # RESEARCH_OOS performance
                ro = df[(df["split"] == "RESEARCH_OOS") & (df["regime_entry"] == regime)]
                emb = df[(df["split"] == "EMBARGO") & (df["regime_entry"] == regime)]
                if len(ro) < 5:
                    continue
                m_ro = apply_conditions(ro, conds)
                m_emb = apply_conditions(emb, conds)
                ro_flagged = ro[m_ro]
                emb_flagged = emb[m_emb]
                ro_n = int(m_ro.sum())
                emb_n = int(m_emb.sum())
                if ro_n < 8:
                    continue
                meta_rule_rows.append({
                    "symbol": sym,
                    "parent_rule_name": rname,
                    "regime": regime,
                    "meta_rule_name": meta_name,
                    "research_oos_n_flagged": ro_n,
                    "research_oos_precision": float(ro_flagged["is_loser"].mean()),
                    "research_oos_net_pnl_impact": float(-ro_flagged["pnl_pct"].sum()),
                    "research_oos_mean_pnl_flagged": float(ro_flagged["pnl_pct"].mean()),
                    "embargo_n_flagged": emb_n,
                    "embargo_precision": float(emb_flagged["is_loser"].mean()) if emb_n else np.nan,
                    "embargo_net_pnl_impact": float(-emb_flagged["pnl_pct"].sum()) if emb_n else 0.0,
                    "embargo_mean_pnl_flagged": float(emb_flagged["pnl_pct"].mean()) if emb_n else np.nan,
                })

    pd.DataFrame(cell_rows).to_csv(OUT / "rule_x_regime_cells.csv", index=False)
    meta = pd.DataFrame(meta_rule_rows).sort_values(["symbol", "research_oos_net_pnl_impact"], ascending=[True, False])
    meta.to_csv(OUT / "regime_conditional_meta_rules.csv", index=False)
    print("\nMeta-rules with positive research-OOS net pnl impact (top):")
    pos = meta[meta["research_oos_net_pnl_impact"] > 0]
    if not pos.empty:
        print(pos[["symbol", "meta_rule_name", "research_oos_n_flagged",
                   "research_oos_precision", "research_oos_net_pnl_impact",
                   "embargo_n_flagged", "embargo_net_pnl_impact"]].head(15).to_string(index=False))
    else:
        print("  none")


if __name__ == "__main__":
    main()
