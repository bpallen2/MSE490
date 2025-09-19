# MSE490
Assignments for MSE490 
## Problem Statement
Current sustainability tools for electronics often present data at the whole-device level and miss opportunities for *component-level exploration*. Educators and learners need an interactive way to mix and match parts (display, circuit boards, casing) from different phones to understand trade‑offs in **cost** and **carbon footprint**. This tool demonstrates how component choices affect the final impact of a phone.

## Acceptance Criteria
- The app must load an EPEAT‑style CSV dataset with at least these columns:
  `Product Name, base_cost_usd, component, kg_co2e, cost_usd`.
- The dataset must include at minimum the components: `display`, `circuit_boards`, `casing`.
- The interface allows users to select one product option for each component.
- The app displays per‑component values and totals for **cost** and **kg CO₂e**.
- A random **goal mode** challenges the user to assemble a phone within a target cost and carbon footprint, with tolerances based on difficulty.
- Sanity checks prevent loading if required columns/components are missing or values are invalid.

## How to Run in Google Colab
1. Open the provided notebook in Google Colab.
2. Upload the dataset `mobile_phone_component_cost_test_dataset.csv` to `/mnt/data/`.
3. Run setup cells:
   - Install Streamlit
   - Save the app script (`sis_components_app.py`)
4. Launch Streamlit via Colab’s proxy — a live link will be displayed and an iframe preview will open.
5. Use the dropdowns to select components and view totals vs. the generated goal.
6. Stop the app by running the “Stop Streamlit” cell when done.

---

*This prototype is for demonstration and educational purposes; it is not an official EPEAT product.*
"""

# Create the directory if it doesn't exist
os.makedirs("/mnt/data/", exist_ok=True)

with open("/mnt/data/README.md", "w") as f:
    f.write(readme_text)

"/mnt/data/README.md written."
