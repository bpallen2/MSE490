from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_sortables import sort_items

# =========================
# Streamlit compatibility helpers
# =========================
def _get_cache_decorator():
    return getattr(st, "cache_data", getattr(st, "cache", None))
_cache = _get_cache_decorator()
if _cache is None:
    def _cache(*args, **kwargs):
        def deco(fn): return fn
        return deco

def qp_get(key: str, default=None):
    qp_obj = getattr(st, "query_params", None)
    if qp_obj is not None:
        return qp_obj.get(key, default)
    old = st.experimental_get_query_params()
    return (old.get(key, [default]) or [default])[0]

def qp_set(**kwargs):
    qp_obj = getattr(st, "query_params", None)
    if qp_obj is not None:
        qp_obj.update({k: str(v) for k, v in kwargs.items()})
    else:
        st.experimental_set_query_params(**{k: str(v) for k, v in kwargs.items()})

def safe_toast(msg: str, icon: str | None = None):
    fn = getattr(st, "toast", None)
    if fn:
        fn(msg, icon=icon)
    else:
        st.info(f"{icon or ''} {msg}")

def safe_progress(value: float, text: str | None = None):
    """Progress bar that works across Streamlit versions."""
    v = max(0.0, min(1.0, float(value)))
    try:
        return st.progress(v, text=text)   # newer Streamlit (float + text)
    except TypeError:
        return st.progress(int(v*100))     # older Streamlit (0-100 int, no text)

# =========================
# Constants
# =========================
REQUIRED_COLS = ["Product Name", "base_cost_usd", "component", "kg_co2e", "cost_usd"]
REQUIRED_COMPONENTS = ["display", "circuit_boards", "casing"]
GITHUB_DATA_URL = "https://raw.githubusercontent.com/bpallen2/MSE490/main/datasets/mobile_phone_component_cost_test_dataset.csv"

# =========================
# Core: load/validate/aggregate
# =========================
@_cache(show_spinner=False)
def load_csv_validated(_: str | None) -> pd.DataFrame:
    df = pd.read_csv(GITHUB_DATA_URL)
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
    return (
        df[df["component"].isin(REQUIRED_COMPONENTS)]
        .groupby(["Product Name", "component"], as_index=False)[["kg_co2e", "cost_usd"]]
        .sum()
        .sort_values(["component", "Product Name"])
        .reset_index(drop=True)
    )

def build_all_options(agg: pd.DataFrame) -> Dict[str, List[str]]:
    return {c: agg.loc[agg["component"] == c, "Product Name"].unique().tolist()
            for c in REQUIRED_COMPONENTS}

def pick_row(agg: pd.DataFrame, component: str, product: str) -> pd.Series:
    row = agg[(agg["component"] == component) & (agg["Product Name"] == product)]
    if row.empty:
        raise ValueError(f"No row for component={component} product={product}")
    return row.iloc[0]

def assemble_totals(agg: pd.DataFrame, selections: Dict[str, str]) -> Dict[str, float | pd.DataFrame]:
    rows = [pick_row(agg, comp, prod) for comp, prod in selections.items()]
    chosen = pd.DataFrame(rows)
    return {
        "total_cost_usd": float(chosen["cost_usd"].sum()),
        "total_kg_co2e": float(chosen["kg_co2e"].sum()),
        "chosen": chosen,
    }

# =========================
# Subset sampling & goal generation (guaranteed achievable)
# =========================
def sample_component_subset(all_options: Dict[str, List[str]], rng: np.random.Generator, k: int = 5) -> Dict[str, List[str]]:
    """Pick up to k unique products per component."""
    subset: Dict[str, List[str]] = {}
    for comp, lst in all_options.items():
        unique = list(dict.fromkeys(lst))
        if len(unique) == 0:
            raise AssertionError(f"No options available for {comp}")
        if len(unique) <= k:
            subset[comp] = unique
        else:
            idxs = rng.choice(len(unique), size=k, replace=False)
            subset[comp] = [unique[i] for i in idxs]
    return subset

def sample_goal_from_subset(agg: pd.DataFrame, subset: Dict[str, List[str]], rng: np.random.Generator) -> Tuple[float,float,Dict[str,str]]:
    picks = {comp: rng.choice(opts) for comp, opts in subset.items()}
    totals = assemble_totals(agg, picks)
    return totals["total_cost_usd"], totals["total_kg_co2e"], picks

def tolerance_for_difficulty(difficulty: str, cost_target: float, carbon_target: float) -> Tuple[float,float]:
    if difficulty == "Easy":
        return max(50.0, 0.10*cost_target), max(5.0, 0.10*carbon_target)
    if difficulty == "Medium":
        return max(30.0, 0.06*cost_target), max(3.0, 0.06*carbon_target)
    return max(15.0, 0.03*cost_target), max(1.5, 0.03*carbon_target)

def within_goal(cost, carbon, goal_cost, goal_carbon, tol_cost, tol_carbon) -> Tuple[bool,bool]:
    return (abs(cost - goal_cost) <= tol_cost, abs(carbon - goal_carbon) <= tol_carbon)

def closeness(val: float, target: float, tol: float) -> float:
    return float(max(0.0, 1.0 - abs(val - target)/max(tol, 1e-9)))

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Phone Component Mixer", page_icon="📱", layout="wide")

st.markdown("""
<style>
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
.hero {
  padding: 1.0rem 1.25rem;
  border-radius: 14px;
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
  color: white;
  box-shadow: 0 10px 24px rgba(0,0,0,0.10);
}
.hero h1 { margin: 0; font-weight: 750; }
.hero p { margin: 0.4rem 0 0; }
[data-testid="stMetric"] {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px; padding: 0.75rem; background: rgba(255,255,255,0.75);
}
footer {visibility: hidden;} #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# Load + validate
# =========================
# Clear the cache before loading data
st.cache_data.clear()
df = load_csv_validated(None)
agg = aggregate_required(df)
all_options = build_all_options(agg)

# =========================
# Goal controls
# =========================
with st.sidebar:
    st.header("🎯 Goal Settings")
    default_diff = qp_get("difficulty", "Medium")
    try:
        default_seed = int(qp_get("seed", 42))
    except Exception:
        default_seed = 42
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"],
                              index=["Easy","Medium","Hard"].index(default_diff) if default_diff in ["Easy","Medium","Hard"] else 1)
    seed = st.number_input("Random seed", min_value=0, value=int(default_seed), step=1)
    gen_btn = st.button("Generate new goal")
    qp_set(difficulty=difficulty, seed=seed)

def reset_practice_state():
    st.session_state.attempts = 0
    st.session_state.revealed = False

def refresh_subset_and_goal(seed_val: int, difficulty: str):
    rng = np.random.default_rng(int(seed_val))
    subset = sample_component_subset(all_options, rng, k=5)
    goal_cost, goal_carbon, goal_pick = sample_goal_from_subset(agg, subset, rng)
    tol_cost, tol_carbon = tolerance_for_difficulty(difficulty, goal_cost, goal_carbon)
    st.session_state.options_subset = subset
    st.session_state.goal = {
        "cost": goal_cost, "carbon": goal_carbon,
        "tol_cost": tol_cost, "tol_carbon": tol_carbon,
        "difficulty": difficulty, "seed": int(seed_val),
        "ref_pick": goal_pick
    }
    # reset DnD & UI state for new subset
    st.session_state["subset_sig"] = None
    st.session_state["sortables_state"] = None
    st.session_state["dnd_sig"] = None
    st.session_state["ui_tick"] = 0
    reset_practice_state()

if "options_subset" not in st.session_state or "goal" not in st.session_state:
    refresh_subset_and_goal(seed, difficulty)
if gen_btn:
    refresh_subset_and_goal(seed, difficulty)

subset = st.session_state.options_subset
goal = st.session_state.goal

# =========================
# Drag-and-Drop: three trays + three slots (with SVG tray badges)
# =========================
st.subheader("Drag parts from the tray into the phone slots")

# Build per-category trays
EMOJI = {"display": "📱", "circuit_boards": "🔌", "casing": "🧩"}
def make_label(comp: str, prod: str) -> str:
    return f"{EMOJI[comp]} {prod}"

tray_items_by_comp = {c: [] for c in REQUIRED_COMPONENTS}
lookup = {}
for comp in REQUIRED_COMPONENTS:
    for opt in subset[comp]:
        lbl = make_label(comp, opt)
        tray_items_by_comp[comp].append(lbl)
        lookup[lbl] = (comp, opt)

# Reset sortables state if subset changed
subset_sig = tuple((c, tuple(subset[c])) for c in REQUIRED_COMPONENTS)
if st.session_state.get("subset_sig") != subset_sig:
    st.session_state["subset_sig"] = subset_sig
    st.session_state["sortables_state"] = None
    st.session_state["dnd_sig"] = None
    st.session_state["ui_tick"] = 0

# SVGs and badge CSS for tray headers
SVG_DISPLAY = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128' width='22' height='22'>
  <rect x='20' y='8' width='88' height='112' rx='14' ry='14' fill='#111827'/>
  <rect x='26' y='20' width='76' height='88' rx='8' ry='8' fill='#3B82F6' stroke='#0EA5E9' stroke-width='2'/>
</svg>"""
SVG_BOARD = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128' width='22' height='22'>
  <rect x='8' y='12' width='112' height='104' rx='10' fill='#065F46' stroke='#10B981' stroke-width='3'/>
  <rect x='30' y='38' width='20' height='16' rx='2' fill='#111827'/>
</svg>"""
SVG_CASE = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128' width='22' height='22'>
  <rect x='24' y='8' width='80' height='112' rx='18' ry='18' fill='#F59E0B' stroke='#B45309' stroke-width='3'/>
</svg>"""

st.markdown("""
<style>
.badge {display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .6rem;font-size:.85rem;font-weight:600;
        color:#111;background:#f3f4f6;border-radius:12px;margin-bottom:.35rem;border:1px solid #e5e7eb;}
.badge.blue   { background:#e8f2ff; border-color:#cfe3ff; }
.badge.green  { background:#e6fbf4; border-color:#c6f7e9; }
.badge.orange { background:#fff4e5; border-color:#ffe1bf; }
</style>
""", unsafe_allow_html=True)

hdr = st.columns(3)
with hdr[0]:
    st.markdown(f"<div class='badge blue'>{SVG_DISPLAY} Display Tray</div>", unsafe_allow_html=True)
with hdr[1]:
    st.markdown(f"<div class='badge green'>{SVG_BOARD} Circuit Board Tray</div>", unsafe_allow_html=True)
with hdr[2]:
    st.markdown(f"<div class='badge orange'>{SVG_CASE} Casing Tray</div>", unsafe_allow_html=True)

# Define initial containers: 3 trays + 3 slots
initial = [
    {"header": "🧰 Display Tray",       "items": tray_items_by_comp["display"]},
    {"header": "🧰 Circuit Board Tray", "items": tray_items_by_comp["circuit_boards"]},
    {"header": "🧰 Casing Tray",        "items": tray_items_by_comp["casing"]},
    {"header": "📱 Display Slot",       "items": []},
    {"header": "🔌 Circuit Board Slot", "items": []},
    {"header": "🧩 Casing Slot",        "items": []},
]

base_items = st.session_state.get("sortables_state") or initial

# Render multi-container DnD (guard None on first render)
new_state = sort_items(base_items, multi_containers=True)
if new_state is None:
    new_state = base_items
st.session_state["sortables_state"] = new_state

# Normalize items from sortables (strings or dicts)
def _label_from_item(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("label", "text", "content", "value"):
            if k in x and isinstance(x[k], str):
                return x[k]
    return str(x)

by_header = {}
for container in new_state:
    header = container.get("header", "")
    items_norm = [_label_from_item(i) for i in (container.get("items") or [])]
    by_header[header] = items_norm

# Bump UI tick if layout changed → ensures progress bars redraw
dnd_sig = tuple((h, tuple(by_header[h])) for h in sorted(by_header.keys()))
if st.session_state.get("dnd_sig") != dnd_sig:
    st.session_state["dnd_sig"] = dnd_sig
    st.session_state["ui_tick"] = st.session_state.get("ui_tick", 0) + 1

# Parse selections: take the first card in each slot
selections: Dict[str, str] = {}
for comp, header in [
    ("display",        "📱 Display Slot"),
    ("circuit_boards", "🔌 Circuit Board Slot"),
    ("casing",         "🧩 Casing Slot"),
]:
    items_in_slot = by_header.get(header, []) or []
    if items_in_slot:
        lbl = items_in_slot[0]
        if lbl in lookup:
            _, prod = lookup[lbl]
            selections[comp] = prod

# Fallback if a slot is empty
for comp in REQUIRED_COMPONENTS:
    if comp not in selections and subset[comp]:
        selections[comp] = subset[comp][0]

# =========================
# Goal display
# =========================
st.subheader("🎯 Your Goal")
gc1, gc2, gc3 = st.columns([1,1,2])
with gc1: st.metric("Target cost (USD)", f"${goal['cost']:,.2f}")
with gc2: st.metric("Target carbon (kg CO₂e)", f"{goal['carbon']:.2f}")
with gc3: st.write(f"Tolerance → cost: ±${goal['tol_cost']:.2f}, carbon: ±{goal['tol_carbon']:.2f}  |  Difficulty: **{goal['difficulty']}**")

with st.expander("What generated this goal? (hidden until 3 fails)"):
    if st.session_state.get("revealed", False):
        st.json({"reference_combo": goal["ref_pick"], "seed": goal["seed"], "subset": subset})
    else:
        st.caption("Keep trying! The reference combo & subset appear after 3 failed attempts.")

# =========================
# Live Totals (auto-update on drag)
# =========================
totals = assemble_totals(agg, selections)
chosen = totals["chosen"].rename(columns={"Product Name": "product"})

st.subheader("📊 Your Build Totals")
colA, colB = st.columns(2)
with colA:
    st.metric("💵 Total Cost (USD)",
              f"${totals['total_cost_usd']:,.2f}",
              delta=f"{totals['total_cost_usd'] - goal['cost']:+.2f}")
with colB:
    st.metric("🌍 Total Carbon (kg CO₂e)",
              f"{totals['total_kg_co2e']:.2f}",
              delta=f"{totals['total_kg_co2e'] - goal['carbon']:+.2f}")

# Closeness bars with fresh placeholders (reliable redraw)
cost_close = closeness(totals["total_cost_usd"], goal["cost"], goal["tol_cost"])
carbon_close = closeness(totals["total_kg_co2e"], goal["carbon"], goal["tol_carbon"])
tick = st.session_state.get("ui_tick", 0)

c1, c2 = st.columns(2)
with c1:
    st.caption(f"Cost closeness — {cost_close*100:.1f}%")
    ph_cost = st.empty()
    try:
        ph_cost.progress(cost_close, text=f"{cost_close*100:.1f}% within tolerance")
    except TypeError:
        ph_cost.progress(int(cost_close*100))
with c2:
    st.caption(f"Carbon closeness — {carbon_close*100:.1f}%")
    ph_carbon = st.empty()
    try:
        ph_carbon.progress(carbon_close, text=f"{carbon_close*100:.1f}% within tolerance")
    except TypeError:
        ph_carbon.progress(int(carbon_close*100))

gap_cost = totals["total_cost_usd"] - goal["cost"]
gap_co2 = totals["total_kg_co2e"] - goal["carbon"]
st.caption(f"Δ Cost: {gap_cost:+.2f} (±{goal['tol_cost']:.2f})  •  Δ Carbon: {gap_co2:+.2f} (±{goal['tol_carbon']:.2f})")

# Detail table
st.dataframe(chosen[["product","component","kg_co2e","cost_usd"]], use_container_width=True)

# =========================
# Practice Mode (button-based check)
# =========================
st.markdown("### Practice Mode")
colA, colB, colC, colD = st.columns([1,1,1,2])
with colA: check_btn = st.button("Check attempt ✅")
with colB: reset_btn = st.button("Reset attempts 🔄")
with colC: reveal_btn = st.button("Reveal now 👀")
with colD:
    attempts = int(st.session_state.get("attempts", 0))
    safe_progress(min(attempts, 3)/3.0, text=f"Attempts: {attempts}/3")

if reset_btn:
    reset_practice_state(); safe_toast("Attempts reset.", icon="↩️")
if reveal_btn:
    st.session_state.revealed = True; safe_toast("Reference combo revealed below.", icon="👀")

if check_btn:
    ok_cost, ok_carbon = within_goal(totals["total_cost_usd"], totals["total_kg_co2e"],
                                     goal["cost"], goal["carbon"],
                                     goal["tol_cost"], goal["tol_carbon"])
    if ok_cost and ok_carbon:
        st.success("✅ You hit the goal! Both cost and carbon are within tolerance.")
        st.balloons()
        safe_toast("Nice shot! 🎉", icon="🎯")
        reset_practice_state()
    else:
        st.session_state.attempts += 1
        msgs = []
        if not ok_cost:
            msgs.append(f"Cost off by {gap_cost:+.2f} (±{goal['tol_cost']:.2f})")
        if not ok_carbon:
            msgs.append(f"Carbon off by {gap_co2:+.2f} (±{goal['tol_carbon']:.2f})")
        st.warning("Attempt not within tolerance:\n- " + "\n- ".join(msgs))
        safe_toast("Keep iterating ⚙️", icon="🧭")
        if st.session_state.attempts >= 3 and not st.session_state.revealed:
            st.session_state.revealed = True
            st.info("👀 Reference combo revealed below.")
            st.experimental_rerun()

# =========================
# Download current build
# =========================
out = chosen[["product","component","kg_co2e","cost_usd"]].copy()
out.loc[:, "total_cost_usd"] = totals["total_cost_usd"]
out.loc[:, "total_kg_co2e"] = totals["total_kg_co2e"]
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download current build (CSV)", data=csv,
                   file_name="phone_mix.csv", mime="text/csv")

st.write("")
st.markdown("""<div style="text-align:center;opacity:0.7;padding-top:0.5rem;">
  Made with ❤️ for classroom exploration • Streamlit prototype
</div>""", unsafe_allow_html=True)
