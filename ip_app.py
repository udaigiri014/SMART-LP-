import streamlit as st
import pulp

# Try importing pandas (for Excel upload feature)
try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# ---------------- Streamlit Page Settings ----------------
st.set_page_config(page_title="LP Solver", layout="wide")

# ---------------- Add CSS for background and button styling ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&display=swap');
    
    /* Apply to all Streamlit headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lora', serif;
        color: #081f5c;
    }
    
    /* Specific header customizations */
    h1 {
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Keep your existing styles */
    .stApp {
        background-color: #f7f2eb;
        font-family: 'Segoe UI', sans-serif;
    }
    .description {
        font-size: 18px;
        margin-top: 10px;
        color: #333333;
        line-height: 1.5;
    }
    .stButton>button {
        background-color: #081f5c;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        padding: 0.5em 1.2em;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0a2b7a;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f2eb;
        font-family: 'Segoe UI', sans-serif;
    }
    .description {
        font-size: 18px;
        margin-top: 10px;
        color: #333333;
        line-height: 1.5;
    }
    .stButton>button {
        background-color: #081f5c;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        padding: 0.5em 1.2em;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0a2b7a;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #081f5c;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# LP MODELS: Three optimization problem solvers using PuLP
# ------------------------------------------------------------

def transportation_problem(supply, demand, cost, capacity):
    """Transportation Problem: minimize total transport cost under supply/demand and capacity."""
    # Define LP problem
    prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

    # Create variables for each (supply point, demand point) route
    routes = [(s, d) for s in supply for d in demand]
    x = pulp.LpVariable.dicts("Route", routes, lowBound=0, cat='Continuous')

    # Objective: Minimize total cost = sum(cost * shipped quantity)
    prob += pulp.lpSum([cost[s][d] * x[(s, d)] for (s, d) in routes])

    # Constraint: Do not exceed supply at each source
    for s in supply:
        prob += pulp.lpSum([x[(s, d)] for d in demand]) <= supply[s]

    # Constraint: Meet demand at each destination
    for d in demand:
        prob += pulp.lpSum([x[(s, d)] for s in supply]) >= demand[d]

    # Optional constraint: Road capacity if provided
    for (s, d) in routes:
        cap_val = capacity.get((s, d), None)
        if cap_val is not None and cap_val != float("inf"):
            prob += x[(s, d)] <= cap_val

    # Solve LP
    prob.solve()

    # Return allocation for each route and total cost
    return {(s, d): x[(s, d)].varValue for (s, d) in routes}, pulp.value(prob.objective)


def vehicle_routing(customers, distances, vehicle_capacity):
    """Vehicle Routing Problem: Simplified single-vehicle version using MTZ constraints."""
    # Define LP problem
    prob = pulp.LpProblem("Vehicle_Routing", pulp.LpMinimize)
    nodes = list(customers.keys())
    routes = [(i, j) for i in nodes for j in nodes if i != j]

    # Variables: x[(i,j)] = 1 if route used, else 0
    x = pulp.LpVariable.dicts("Route", routes, cat="Binary")
    # Load variables for subtour elimination
    u = pulp.LpVariable.dicts("Load", nodes, lowBound=0)

    # Objective: minimize total distance
    prob += pulp.lpSum([distances[i][j] * x[(i, j)] for (i, j) in routes])

    # Each customer visited exactly once
    for k in nodes[1:]:
        prob += pulp.lpSum([x[(i, k)] for i in nodes if i != k]) == 1
        prob += pulp.lpSum([x[(k, j)] for j in nodes if j != k]) == 1

    # Subtour elimination (MTZ)
    for i in nodes[1:]:
        for j in nodes[1:]:
            if i != j:
                prob += u[i] - u[j] + vehicle_capacity * x[(i, j)] <= vehicle_capacity - customers[j]

    for i in nodes[1:]:
        prob += u[i] >= customers[i]
        prob += u[i] <= vehicle_capacity

    prob.solve()
    return [(i, j) for (i, j) in routes if x[(i, j)].varValue == 1], pulp.value(prob.objective)


def facility_location(fixed_costs, shipping_costs, demand, supply_limit):
    """Facility Location: Decide which facilities to open and how to allocate customers."""
    # Define LP problem
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    facilities = list(fixed_costs.keys())
    customers = list(demand.keys())

    # Variables: shipment quantities and facility open/closed status
    x = pulp.LpVariable.dicts("Ship", [(f, c) for f in facilities for c in customers], lowBound=0)
    y = pulp.LpVariable.dicts("Open", facilities, cat='Binary')

    # Objective: Minimize fixed + shipping costs
    prob += (
        pulp.lpSum([fixed_costs[f] * y[f] for f in facilities]) +
        pulp.lpSum([shipping_costs[f][c] * x[(f, c)] for f in facilities for c in customers])
    )

    # Constraints: meet demand, do not exceed facility capacity if closed
    for c in customers:
        prob += pulp.lpSum([x[(f, c)] for f in facilities]) >= demand[c]
    for f in facilities:
        prob += pulp.lpSum([x[(f, c)] for c in customers]) <= supply_limit[f] * y[f]

    prob.solve()

    # Return open facilities, assignments, and total cost
    open_facilities = [f for f in facilities if y[f].varValue == 1]
    routes = {(f, c): x[(f, c)].varValue for f in facilities for c in customers}
    return open_facilities, routes, pulp.value(prob.objective)


# ------------------------------------------------------------
# Parse Excel for Transportation Problem
# ------------------------------------------------------------
def parse_transport_excel(uploaded_file):
    """Read Excel with costs, supply, demand for transportation problem."""
    if not PANDAS_OK:
        raise ImportError("pandas not available; cannot parse Excel.")

    df = pd.read_excel(uploaded_file, index_col=0)

    # Normalize labels
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    # Identify Demand row and Supply column
    demand_row = next((idx for idx in df.index if idx.lower().startswith("demand")), None)
    supply_col = next((col for col in df.columns if col.lower().startswith("supply")), None)

    if demand_row is None or supply_col is None:
        raise ValueError("Excel must have 'Demand' row and 'Supply' column.")

    # Extract supply, demand, and cost matrix
    supply_rows = [r for r in df.index if r != demand_row]
    demand_cols = [c for c in df.columns if c != supply_col]
    supply = {r: float(df.loc[r, supply_col]) for r in supply_rows}
    demand = {c: float(df.loc[demand_row, c]) for c in demand_cols}
    cost = {s: {d: float(df.loc[s, d]) for d in demand_cols} for s in supply_rows}

    # Default: no hard capacity limits
    capacity = {(s, d): float("inf") for s in supply for d in demand}
    return supply, demand, cost, capacity


# ------------------------------------------------------------
# Navigation Functions
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home(): st.session_state.page = "home"
def go_transport(): st.session_state.page = "transport"
def go_routing(): st.session_state.page = "routing"
def go_facility(): st.session_state.page = "facility"


# ------------------------------------------------------------
# Page Routing: Home | Transportation | Routing | Facility
# ------------------------------------------------------------
page = st.session_state.page

if page == "home":
    # Create outer columns to center the content
    left_pad, center_col, right_pad = st.columns([1, 6, 1])
    
    with center_col:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Sorts+Mill+Goudy:ital@0;1&display=swap');
        
        .lovelace-title {
            font-family: 'Lora', 'Sorts Mill Goudy', 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
            font-weight: 700;
            font-size: 3.5rem;
            text-align: center;
            color: #081f5c;
            margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .lovelace-subtitle {
            font-family: 'Lora', 'Sorts Mill Goudy', 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
            font-size: 1.5rem;
            text-align: center;
            color: #333333;
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.4;
        }
        .goudy-text {
            font-family: 'Sorts Mill Goudy', Georgia, 'Times New Roman', Times, serif;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .divider {
            border-top: 2px solid #081f5c;
            opacity: 0.2;
            margin: 1.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="lovelace-title">Smart LP</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="lovelace-subtitle">A smart solution for solving real-world transportation and logistics problems</h3>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        # Create 3 equal columns for the buttons and descriptions
        col1, col2, col3 = st.columns(3)
        
        # Transportation Section
        with col1:
            st.button(" Transportation", on_click=go_transport)
            st.markdown(
                """
                <div class='description goudy-text'>
                Find the most cost-effective way to transport goods from suppliers to destinations 
                while meeting supply and demand.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Vehicle Routing Section
        with col2:
            st.button(" Vehicle Routing", on_click=go_routing)
            st.markdown(
                """
                <div class='description goudy-text'>
                Plan the shortest and most efficient delivery routes for vehicles to serve all 
                customers without exceeding capacity.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Facility Location Section
        with col3:
            st.button(" Facility Location", on_click=go_facility)
            st.markdown(
                """
                <div class='description goudy-text'>
                Decide which facilities to open and how to serve customers at minimal cost.
                </div>
                """,
                unsafe_allow_html=True
            )

elif page == "transport":
    st.header(" Transportation Problem")
    st.button("Back to Home", on_click=go_home)

    # Option 1: Excel Upload
    uploaded_file = st.file_uploader("Upload Transportation Matrix (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            supply, demand, cost, capacity = parse_transport_excel(uploaded_file)
            st.write("**Supply:**", supply)
            st.write("**Demand:**", demand)
            st.write("**Cost Matrix:**", cost)
            if st.button("🧠 Solve Using Uploaded Data"):
                res, total = transportation_problem(supply, demand, cost, capacity)
                st.success(f"✅ Total Cost: {total}")
                for (s, d), val in res.items():
                    st.write(f"{s} → {d}: {val:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Or Enter Data Manually")
    with st.form("transport_form"):
        ns = st.number_input("Number of Supply Points", 1, 5, 2)
        nd = st.number_input("Number of Demand Points", 1, 5, 2)
        supply = {f"S{i+1}": st.number_input(f"Supply S{i+1}", 0) for i in range(ns)}
        demand = {f"D{j+1}": st.number_input(f"Demand D{j+1}", 0) for j in range(nd)}
        cost = {s: {d: st.number_input(f"Cost {s} → {d}", 0.0) for d in demand} for s in supply}
        cap = {(s, d): st.number_input(f"Capacity {s} → {d}", 0) for s in supply for d in demand}
        submitted = st.form_submit_button("🧠 Solve")
        if submitted:
            res, total = transportation_problem(supply, demand, cost, cap)
            st.success(f"✅ Total Cost: {total}")
            for (s, d), val in res.items():
                st.write(f"{s} → {d}: {val:.2f}")

elif page == "routing":
    st.header(" Vehicle Routing Problem")
    st.button("Back to Home", on_click=go_home)
    # Input form for VRP
    with st.form("vrp_form"):
        n = st.number_input("Number of Locations (incl. depot)", 2, 6, 4)
        cust = {i: st.number_input(f"Demand at {i}", 0) for i in range(n)}
        dist = {i: {j: st.number_input(f"Distance {i} → {j}", 0.0) for j in range(n) if i != j} for i in range(n)}
        cap = st.number_input("Vehicle Capacity", 1, 100, 40)
        submitted = st.form_submit_button("🚀 Solve")
        if submitted:
            route, total_dist = vehicle_routing(cust, dist, cap)
            st.success(f"✅ Total Distance: {total_dist}")
            for i, j in route:
                st.write(f"{i} → {j}")

elif page == "facility":
    st.header(" Facility Location Problem")
    st.button("Back to Home", on_click=go_home)
    # Input form for Facility Location
    with st.form("fac_form"):
        nf = st.number_input("Number of Facilities", 1, 5, 2)
        nc = st.number_input("Number of Customers", 1, 5, 2)
        fix_cost = {f"F{i+1}": st.number_input(f"Fixed Cost F{i+1}", 0) for i in range(nf)}
        demand = {f"C{j+1}": st.number_input(f"Demand C{j+1}", 0) for j in range(nc)}
        limit = {f"F{i+1}": st.number_input(f"Supply Limit F{i+1}", 0) for i in range(nf)}
        ship_cost = {f: {c: st.number_input(f"Shipping {f} → {c}", 0.0) for c in demand} for f in fix_cost}
        submitted = st.form_submit_button("🏁 Solve")
        if submitted:
            open_facs, routes, total_cost = facility_location(fix_cost, ship_cost, demand, limit)
            st.success(f"✅ Total Cost: {total_cost}")
            st.write("### Open Facilities")
            st.write(open_facs)
            st.write("### Assignments")
            for (f, c), val in routes.items():
                st.write(f"{f} → {c}: {val:.2f}")
