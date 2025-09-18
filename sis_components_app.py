from __future__ import annotations
import math, random
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st

REQUIRED_COLS = ["Product Name", "base_cost_usd", "component", "kg_co2e", "cost_usd"]
REQUIRED_COMPONENTS = ["display", "circuit_boards", "casing"]

# ---------- core: load/validate/aggregate ----------
def load_csv_validated(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"CSV missing required columns: {missing}"
    df["component"] = df["component"].astype(str).str.strip()
    present = sorted(df["component"].dropna().unique().tolist())
    missing_comps = [c for c in REQUIRED_COMPONENTS if c not in present]
    assert not missing_comps, f"Required components not found: {missing_comps}. Present={present}"
    for col in ["kg_co2e", "cost_usd", "base_cost_usd"]:
        bad = ~pd.to_numeric(df[col], errors="coerce").notna()
        assert not bad.any(), f"Non-numeric values in {col}"
        assert (df[col] >= 0).all(), f"{col} must be >= 0"
    return df

def aggregate_required(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["component"].isin(REQUIRED_COMPONENTS)].copy()
    agg = (
        df.groupby(["Product Name", "component"], as_index=False)[["kg_co2e", "cost_usd"]]
          .sum()
          .sort_values(["component", "Product Name"])
          .reset_index(drop=True)
    )
    dups = agg.duplicated(["Product Name", "component"]).sum()
    assert dups == 0, f"Unexpected duplicates after aggregation: {dups}"
    return agg

def build_options(agg: pd.DataFrame) -> Dict[str, List[str]]:
    return {c: agg.loc[agg["component"] == c, "Product Name"].unique().tolist()
            for c in REQUIRED_COMPONENTS}

def pick_row(agg: pd.DataFrame, component: str, product: str) -> pd.Series:
    row = agg[(agg["component"] == component) & (agg["Product Name"] == product)]
    if row.empty:
        raise ValueError(f"No row for component={component} product={product}")
    return row.iloc[0]

def assemble_totals(agg: pd.DataFrame, selections: Dict[str, str]) -> Dict[str, float]:
    rows: List[pd.Series] = []
    for comp, prod in selections.items():
        rows.append(pick_row(agg, comp, prod))
    chosen = pd.DataFrame(rows)
    total_cost = float(chosen["cost_usd"].sum())
    total_kg = float(chosen["kg_co2e"].sum())
    assert total_cost >= 0 and total_kg >= 0
    return {"total_cost_usd": total_cost, "total_kg_co2e": total_kg, "chosen": chosen}

# ---------- goal generation ----------
def sample_random_goal(agg: pd.DataFrame, rng: np.random.Generator) -> Tuple[float, float, Dict[str,str]]:
    selections = {}
    for comp in REQUIRED_COMPONENTS:
        opts = agg.loc[agg["component"] == comp, "Product Name"].unique().tolist()
        if not opts:
            raise AssertionError(f"No options available for {comp}")
        selections[comp] = rng.choice(opts)
    totals = assemble_totals(agg, selections)
    return totals["total_cost_usd"], totals["total_kg_co2e"], selections

def tolerance_for_difficulty(difficulty: str, cost_target: float, carbon_target: float) -> Tuple[float,float]:
    if difficulty == "Easy":
        return max(50.0, 0.10*cost_target), max(5.0, 0.10*carbon_target)
    if difficulty == "Medium":
        return max(30.0, 0.06*cost_target), max(3.0, 0.06*carbon_target)
    return max(15.0, 0.03*cost_target), max(1.5, 0.03*carbon_target)

def within_goal(cost, carbon, goal_cost, goal_carbon, tol_cost, tol_carbon) -> Tuple[bool,bool]:
    return (abs(cost - goal_cost) <= tol_cost, abs(carbon - goal_carbon) <= tol_carbon)

# ---------- UI ----------
st.set_page_config(page_title="Phone Component Mixer (EPEAT CSV)", page_icon="📱", layout="centered")
st.title("📱 Phone Component Mixer — EPEAT CSV + 🎯 Goal Mode (Practice)")

with st.sidebar:
    st.header("Data")
    csv_path = st.text_input(
        "CSV path",
        value="/content/mobile_phone_component_cost_test_dataset.csv", # Corrected path
        help="Path to your uploaded CSV."
    )
    st.caption("Required cols: " + ", ".join(REQUIRED_COLS))
    st.caption("Required components: " + ", ".join(REQUIRED_COMPONENTS))

# Load + validate
try:
    df = load_csv_validated(csv_path)
    agg = aggregate_required(df)
    options = build_options(agg)
    empty = [c for c, lst in options.items() if len(lst) == 0]
    assert not empty, f"No product options available for: {empty}"
    st.success("CSV loaded and validated ✅")
except AssertionError as e:
    st.error(f"Sanity check failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# ----- Goal controls -----
with st.sidebar:
    st.header("🎯 Goal Settings")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    gen_btn = st.button("Generate new goal")
    st.caption("Practice Mode: Reveal reference combo after 3 failed attempts.")

# Initialize/refresh session state for attempts & goal
def reset_practice_state():
    st.session_state.attempts = 0
    st.session_state.revealed = False

if "attempts" not in st.session_state:
    reset_practice_state()

if "goal" not in st.session_state or gen_btn:
    rng = np.random.default_rng(int(seed))
    goal_cost, goal_carbon, goal_pick = sample_random_goal(agg, rng)
    tol_cost, tol_carbon = tolerance_for_difficulty(difficulty, goal_cost, goal_carbon)
    st.session_state.goal = {
        "cost": goal_cost, "carbon": goal_carbon,
        "tol_cost": tol_cost, "tol_carbon": tol_carbon,
        "difficulty": difficulty, "seed": int(seed),
        "ref_pick": goal_pick
    }
    reset_practice_state()

goal = st.session_state.goal

st.subheader("🎯 Your Goal")
gc1, gc2, gc3 = st.columns([1,1,2])
with gc1:
    st.metric("Target cost (USD)", f"${goal['cost']:,.2f}")
with gc2:
    st.metric("Target carbon (kg CO₂e)", f"{goal['carbon']:.2f}")
with gc3:
    st.write(f"Tolerance → cost: ±${goal['tol_cost']:.2f}, carbon: ±{goal['tol_carbon']:.2f}  |  Difficulty: **{goal['difficulty']}**")
with st.expander("What generated this goal? (hidden until 3 fails)"):
    if st.session_state.revealed:
        st.json({"reference_combo": goal["ref_pick"], "seed": goal["seed"]})
    else:
        st.caption("Keep trying! The reference combo appears after 3 failed attempts.")

# ----- Selection UI -----
st.subheader("Select components (mix & match across phones)")
cols = st.columns(3)
selections: Dict[str, str] = {}
for i, comp in enumerate(REQUIRED_COMPONENTS):
    with cols[i]:
        opts = options[comp]
        default_index = 0 if len(opts) > 0 else None
        selections[comp] = st.selectbox(
            f"{comp.replace('_',' ').title()}",
            opts, index=default_index,
            help=f"Choose which phone's {comp.replace('_',' ')} to use."
        )

# Compute current totals (live view)
totals = assemble_totals(agg, selections)
chosen = totals["chosen"].rename(columns={"Product Name": "product"})
st.subheader("Your assembled phone (selected components)")
st.dataframe(chosen[["product","component","kg_co2e","cost_usd"]], use_container_width=True)

st.markdown("### Totals vs Goal")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.metric(
        "Total cost (USD)",
        f"${totals['total_cost_usd']:,.2f}",
        delta=f"{totals['total_cost_usd'] - goal['cost']:+.2f}"
    )
with c2:
    st.metric(
        "Total carbon (kg CO₂e)",
        f"{totals['total_kg_co2e']:.2f}",
        delta=f"{totals['total_kg_co2e'] - goal['carbon']:+.2f}"
    )

# ----- Practice Mode controls -----
st.markdown("### Practice Mode")
colA, colB, colC = st.columns([1,1,1])
with colA:
    check_btn = st.button("Check attempt ✅")
with colB:
    reset_btn = st.button("Reset attempts 🔄")
with colC:
    reveal_btn = st.button("Reveal now 👀")

if reset_btn:
    reset_practice_state()
    st.info("Attempts reset.")

if reveal_btn:
    st.session_state.revealed = True

# Evaluate attempt only when user clicks the button
if check_btn:
    ok_cost, ok_carbon = within_goal(
        totals["total_cost_usd"], totals["total_kg_co2e"],
        goal["cost"], goal["carbon"],
        goal["tol_cost"], goal["tol_carbon"]
    )
    if ok_cost and ok_carbon:
        st.success("✅ You hit the goal! Both cost and carbon are within tolerance.")
        reset_practice_state()
    else:
        st.session_state.attempts += 1
        msgs = []
        if not ok_cost:
            gap = totals["total_cost_usd"] - goal["cost"]
            msgs.append(f"Cost off by {gap:+.2f} (tolerance ±{goal['tol_cost']:.2f})")
        if not ok_carbon:
            gap = totals["total_kg_co2e"] - goal["carbon"]
            msgs.append(f"Carbon off by {gap:+.2f} (tolerance ±{goal['tol_carbon']:.2f})")
        st.warning("Attempt not within tolerance:\n- " + "\n- ".join(msgs))
        st.info(f"Attempts: {st.session_state.attempts}/3")
        if st.session_state.attempts >= 3 and not st.session_state.revealed:
            st.session_state.revealed = True
            st.info("👀 Reference combo revealed below.")
            st.experimental_rerun()

with st.expander("Derived metrics"):
    det = chosen.copy()
    det["cost_per_kg_usd_selected"] = det.apply(
        lambda r: (r["cost_usd"] / r["kg_co2e"]) if r["kg_co2e"] > 0 else float("nan"),
        axis=1
    )
    st.dataframe(det, use_container_width=True)
    st.caption("Note: cost-per-kg is undefined when kg_CO₂e is 0.")
