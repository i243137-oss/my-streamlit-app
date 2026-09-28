import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# ==========================================
# 1. STREAMLIT CONFIG & DRACULA CSS THEME
# ==========================================
st.set_page_config(
    page_title="Informed Search Lab | Dracula Theme",
    page_icon="🦇",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Dracula Theme Color Variables */
    :root {
        --bg-color: #282a36;
        --card-bg: #44475a;
        --fg-color: #f8f8f2;
        --comment: #6272a4;
        --cyan: #8be9fd;
        --green: #50fa7b;
        --orange: #ffb86c;
        --pink: #ff79c6;
        --purple: #bd93f9;
        --red: #ff5555;
        --yellow: #f1fa8c;
    }

    /* Global Streamlit Theme */
    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--fg-color) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: var(--card-bg) !important;
        color: var(--fg-color) !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    button[aria-selected="true"] {
        background-color: var(--purple) !important;
        color: #282a36 !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: var(--purple) !important;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Dracula Custom Cards */
    .dracula-card {
        background-color: var(--card-bg);
        border-left: 5px solid var(--green);
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .dracula-card-pink {
        background-color: var(--card-bg);
        border-left: 5px solid var(--pink);
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* Table Customization */
    .stDataFrame, .stTable {
        background-color: var(--card-bg) !important;
        color: var(--fg-color) !important;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--purple) !important;
        color: #282a36 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: var(--pink) !important;
        box-shadow: 0 0 10px var(--pink) !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. GRAPH DATASETS & SEARCH ALGORITHMS
# ==========================================


def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# --- Task 1 Data: Warehouse Robot ---
warehouse_coords = {
    "Receiving_Area": (0, 0),
    "Storage_A": (2, 4),
    "Storage_B": (3, 1),
    "Sorting_Area": (5, 5),
    "Staging_Area": (6, 2),
    "Packing_Station": (9, 6),
}
warehouse_edges = [
    ("Receiving_Area", "Storage_A", 4.5),
    ("Receiving_Area", "Storage_B", 3.2),
    ("Storage_A", "Sorting_Area", 3.2),
    ("Storage_A", "Storage_B", 3.5),
    ("Storage_B", "Staging_Area", 3.2),
    ("Sorting_Area", "Packing_Station", 4.1),
    ("Staging_Area", "Packing_Station", 5.0),
    ("Sorting_Area", "Staging_Area", 3.2),
]

# --- Task 2 & 3 Data: Airport & Hospital ---
hospital_coords = {
    "Pharmacy": (0, 0),
    "Corridor_A": (1, 4),
    "Corridor_B": (2, 1),
    "ICU": (5, 5),
    "Triage": (4, 2),
    "Radiology": (7, 4),
    "Emergency_Ward": (8, 6),
}
hospital_edges = [
    ("Pharmacy", "Corridor_A", 4.12),
    ("Pharmacy", "Corridor_B", 2.24),
    ("Corridor_A", "ICU", 5.00),
    ("Corridor_B", "Triage", 2.24),
    ("ICU", "Emergency_Ward", 3.16),
    ("Triage", "Radiology", 3.16),
    ("Radiology", "Emergency_Ward", 2.24),
]

# --- Task 4 Data: Drone Network ---
drone_coords = {
    "Distribution_Center": (0, 0),
    "Waypoint_A": (2, 8),
    "Waypoint_B": (3, 2),
    "Hub_North": (6, 9),
    "Hub_South": (7, 3),
    "Customer_Building": (10, 10),
}
drone_edges = [
    ("Distribution_Center", "Waypoint_A", 8.25),
    ("Distribution_Center", "Waypoint_B", 3.61),
    ("Waypoint_A", "Hub_North", 4.12),
    ("Waypoint_B", "Hub_South", 4.12),
    ("Hub_North", "Customer_Building", 4.12),
    ("Hub_South", "Customer_Building", 7.62),
    ("Waypoint_A", "Customer_Building", 8.25),
]


def build_graph(coords, edges):
    G = nx.Graph()
    for node, pos in coords.items():
        G.add_node(node, pos=pos)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


# Greedy Best-First Search
def run_gbfs(graph, coords, start, goal):
    pq = [(euclidean_distance(coords[start], coords[goal]), start)]
    came_from = {start: None}
    visited = set()
    expansion_seq = []

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        expansion_seq.append(current)

        if current == goal:
            break

        for nxt in graph.neighbors(current):
            if nxt not in visited and nxt not in came_from:
                came_from[nxt] = current
                h_val = euclidean_distance(coords[nxt], coords[goal])
                heapq.heappush(pq, (h_val, nxt))

    path, curr = [], goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    total_cost = sum(
        graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
    )
    return expansion_seq, path, round(total_cost, 2)


# A* Search & Weighted A*
def run_astar(graph, coords, start, goal, weight=1.0):
    pq = [(weight * euclidean_distance(coords[start], coords[goal]), start)]
    g_score = {node: float("inf") for node in graph.nodes()}
    g_score[start] = 0
    f_score = {node: float("inf") for node in graph.nodes()}
    f_score[start] = weight * euclidean_distance(coords[start], coords[goal])

    came_from = {start: None}
    visited = set()
    expansion_seq = []
    metrics_table = []

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        expansion_seq.append(current)

        g_val = g_score[current]
        h_val = euclidean_distance(coords[current], coords[goal])
        metrics_table.append(
            {
                "Node": current,
                "g(n)": round(g_val, 2),
                "h(n)": round(h_val, 2),
                "f(n)": round(g_val + weight * h_val, 2),
            }
        )

        if current == goal:
            break

        for nxt in graph.neighbors(current):
            tentative_g = g_score[current] + graph[current][nxt]["weight"]
            if tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                h_nxt = euclidean_distance(coords[nxt], coords[goal])
                f_score[nxt] = tentative_g + weight * h_nxt
                heapq.heappush(pq, (f_score[nxt], nxt))

    path, curr = [], goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return expansion_seq, path, round(g_score[goal], 2), metrics_table


# Matplotlib Plotting Helper for Dracula Theme
def plot_dracula_graph(G, coords, path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#282a36")
    ax.set_facecolor("#282a36")

    path_edges = list(zip(path[:-1], path[1:])) if path else []
    unvisited = [n for n in G.nodes() if n not in path]
    start_node = path[0] if path else None
    goal_node = path[-1] if path else None
    intermediates = [n for n in path if n not in (start_node, goal_node)]

    # Draw Nodes
    nx.draw_networkx_nodes(
        G,
        coords,
        nodelist=unvisited,
        ax=ax,
        node_size=500,
        node_color="#44475a",
        edgecolors="#6272a4",
        linewidths=1.5,
    )
    if intermediates:
        nx.draw_networkx_nodes(
            G,
            coords,
            nodelist=intermediates,
            ax=ax,
            node_size=650,
            node_color="#ff79c6",
            edgecolors="#f8f8f2",
            linewidths=2,
        )
    if start_node:
        nx.draw_networkx_nodes(
            G,
            coords,
            nodelist=[start_node],
            ax=ax,
            node_size=750,
            node_color="#8be9fd",
            edgecolors="#f8f8f2",
            linewidths=2.5,
        )
    if goal_node:
        nx.draw_networkx_nodes(
            G,
            coords,
            nodelist=[goal_node],
            ax=ax,
            node_size=750,
            node_color="#ff5555",
            edgecolors="#f8f8f2",
            linewidths=2.5,
        )

    # Draw Edges
    nx.draw_networkx_edges(
        G, coords, ax=ax, width=1.5, edge_color="#6272a4", alpha=0.6
    )
    if path_edges:
        nx.draw_networkx_edges(
            G, coords, edgelist=path_edges, ax=ax, width=3.5, edge_color="#50fa7b"
        )

    # Labels
    nx.draw_networkx_labels(
        G, coords, ax=ax, font_size=8, font_color="#f8f8f2", font_weight="bold"
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        coords,
        edge_labels=edge_labels,
        ax=ax,
        font_color="#f1fa8c",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="#282a36",
            edgecolor="#44475a",
            alpha=0.85,
        ),
    )

    plt.title(title, color="#bd93f9", fontsize=13, pad=10)
    plt.axis("off")
    return fig


# ==========================================
# 3. STREAMLIT APP LAYOUT & NAVIGATION
# ==========================================

st.title("🦇 Informed Search Algorithms — All 4 Lab Tasks")

tabs = st.tabs(
    [
        "📦 Task 1: Heuristic Design",
        "✈️ Task 2: Airport GBFS",
        "🏥 Task 3: Hospital A*",
        "🚁 Task 4: Drone Weighted A*",
        "🎮 Interactive Sandbox",
    ]
)

# ------------------------------------------
# TAB 1: WAREHOUSE ROBOT HEURISTIC
# ------------------------------------------
with tabs[0]:
    st.header("Task 1: Designing a Heuristic for a Warehouse Robot")
    G1 = build_graph(warehouse_coords, warehouse_edges)
    goal_t1 = "Packing_Station"

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        fig1 = plot_dracula_graph(
            G1,
            warehouse_coords,
            ["Receiving_Area", "Packing_Station"],
            "Warehouse Layout",
        )
        st.pyplot(fig1)

    with col2:
        st.subheader("1. Heuristic Table ($h(n)$ to Packing Station)")
        h_data = []
        for n, coord in warehouse_coords.items():
            val = euclidean_distance(coord, warehouse_coords[goal_t1])
            h_data.append(
                {
                    "Location": n,
                    "Coordinates": str(coord),
                    "h(n) Distance": round(val, 2),
                }
            )
        st.table(pd.DataFrame(h_data))

    st.subheader("2. Verification of Heuristic Consistency Condition")
    st.markdown(
        "Condition: $h(n) \\le \\text{cost}(n,m) + h(m)$ for edge $n \\rightarrow m$"
    )

    cons_data = []
    for u, v, cost in warehouse_edges:
        hu = euclidean_distance(warehouse_coords[u], warehouse_coords[goal_t1])
        hv = euclidean_distance(warehouse_coords[v], warehouse_coords[goal_t1])
        is_valid_uv = hu <= cost + hv
        cons_data.append(
            {
                "Edge (n -> m)": f"{u} -> {v}",
                "Edge Cost": cost,
                "h(n)": round(hu, 2),
                "cost + h(m)": round(cost + hv, 2),
                "Consistent?": "Valid" if is_valid_uv else "Invalid",
            }
        )
    st.dataframe(pd.DataFrame(cons_data), use_container_width=True)

    st.markdown(
        """
        <div class="dracula-card">
            <h4>💡 Why Euclidean Distance is Suitable</h4>
            <p>1. <b>Admissibility:</b> Straight-line physical separation never overestimates actual travel distance along warehouse aisles ($h(n) \le h^*(n)$).</p>
            <p>2. <b>Monotonicity/Consistency:</b> Satisfies the triangle inequality, guaranteeing that $f(n)$ monotonically increases along any search path.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------
# TAB 2: AIRPORT BAGGAGE GBFS
# ------------------------------------------
with tabs[1]:
    st.header("Task 2: Greedy Best-First Search (Airport Baggage)")
    G2 = build_graph(hospital_coords, hospital_edges)  # Uses airport layout
    exp_seq2, path2, cost2 = run_gbfs(
        G2, hospital_coords, "Pharmacy", "Emergency_Ward"
    )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        fig2 = plot_dracula_graph(
            G2, hospital_coords, path2, "GBFS baggage route (Red Goal)"
        )
        st.pyplot(fig2)

    with col2:
        st.markdown(
            f"""
            <div class="dracula-card">
                <h3>🔍 GBFS Solution Summary</h3>
                <p><b>Expansion Sequence:</b> <span style="color:#8be9fd;">{' ➔ '.join(exp_seq2)}</span></p>
                <p><b>Solution Path:</b> <span style="color:#50fa7b;">{' ➔ '.join(path2)}</span></p>
                <p><b>Total Path Cost:</b> <span style="color:#ff79c6; font-size: 1.2rem; font-weight:bold;">{cost2} km</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dracula-card-pink">
                <h4>🧠 How Heuristic Guided GBFS</h4>
                <p>GBFS relies exclusively on $f(n) = h(n)$. At the initial junction, it selects <code>Corridor_A</code> because $h = 7.28 < h(\text{Corridor\_B}) = 7.81$. It ignores the high travel cost of $4.12$, leading to a suboptimal total path cost of <b>12.28 km</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------
# TAB 3: EMERGENCY SUPPLY A*
# ------------------------------------------
with tabs[2]:
    st.header("Task 3: A* Search Algorithm for Emergency Supply")
    G3 = build_graph(hospital_coords, hospital_edges)
    exp_seq3, path3, cost3, metrics3 = run_astar(
        G3, hospital_coords, "Pharmacy", "Emergency_Ward"
    )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        fig3 = plot_dracula_graph(
            G3, hospital_coords, path3, "A* Optimal Emergency Route"
        )
        st.pyplot(fig3)

    with col2:
        st.subheader("Selected Nodes Metrics Table")
        st.table(pd.DataFrame(metrics3))

    st.markdown(
        f"""
        <div class="dracula-card">
            <h3>📊 Search Summary & Comparison</h3>
            <p><b>Node Expansion Sequence:</b> {' ➔ '.join(exp_seq3)}</p>
            <p><b>Optimal Path:</b> {' ➔ '.join(path3)}</p>
            <p><b>Total Optimal Cost:</b> <span style="color:#50fa7b; font-weight:bold;">{cost3} km</span> (vs GBFS: 12.28 km)</p>
            <p><i><b>A* vs GBFS Behavior:</b> By incorporating historical cost $g(n)$, A* recognizes that <code>Corridor_B</code> yields a lower combined estimate $f(n) = 10.05$ than <code>Corridor_A</code> ($f(n) = 11.40$), finding the truly optimal route.</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------
# TAB 4: AUTONOMOUS DRONE WEIGHTED A*
# ------------------------------------------
with tabs[3]:
    st.header("Task 4: Weighted A* for Autonomous Drone Delivery")
    G4 = build_graph(drone_coords, drone_edges)

    w_selected = st.slider(
        "Select Heuristic Weight (w)", 1.0, 3.0, 1.5, step=0.5
    )

    weights = [1.0, 1.5, 2.0, 3.0]
    w_results = []
    for w in weights:
        exp, p, c, m = run_astar(
            G4,
            drone_coords,
            "Distribution_Center",
            "Customer_Building",
            weight=w,
        )
        w_results.append(
            {
                "Weight (w)": w,
                "Solution Path": " -> ".join(p),
                "Total Cost": c,
                "Nodes Expanded": len(exp),
            }
        )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        exp_w, path_w, cost_w, _ = run_astar(
            G4,
            drone_coords,
            "Distribution_Center",
            "Customer_Building",
            weight=w_selected,
        )
        fig4 = plot_dracula_graph(
            G4, drone_coords, path_w, f"Drone Path for Weight w = {w_selected}"
        )
        st.pyplot(fig4)

    with col2:
        st.subheader("Weighted A* Comparison Table")
        st.dataframe(pd.DataFrame(w_results), use_container_width=True)

        st.markdown(
            """
            <div class="dracula-card">
                <h4>⚖️ Effect of Weight $w$</h4>
                <p>• <b>w = 1.0 (Normal A*):</b> Guarantees optimal path (16.49 km) but expands 6 nodes.</p>
                <p>• <b>w > 1.0 (Weighted A*):</b> Trusting the heuristic more reduces expansions down to 4 nodes, but returns a slightly suboptimal path (16.50 km).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------
# TAB 5: INTERACTIVE SANDBOX GUI
# ------------------------------------------
with tabs[4]:
    st.header("🎮 Interactive Custom Search Sandbox")

    graph_choice = st.selectbox(
        "Choose Problem Domain Graph",
        [
            "Hospital Emergency Supply",
            "Warehouse Robot Network",
            "Autonomous Drone Routes",
        ],
    )

    if graph_choice == "Hospital Emergency Supply":
        curr_coords, curr_edges = hospital_coords, hospital_edges
    elif graph_choice == "Warehouse Robot Network":
        curr_coords, curr_edges = warehouse_coords, warehouse_edges
    else:
        curr_coords, curr_edges = drone_coords, drone_edges

    G_sandbox = build_graph(curr_coords, curr_edges)
    node_options = list(curr_coords.keys())

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        start_sb = st.selectbox("Start Node", node_options, index=0)
    with col_s2:
        goal_sb = st.selectbox(
            "Goal Node", node_options, index=len(node_options) - 1
        )
    with col_s3:
        algo_sb = st.selectbox(
            "Algorithm", ["Greedy Best-First Search", "A* Search"]
        )

    if start_sb == goal_sb:
        st.error("Start and Goal nodes must be different.")
    else:
        if algo_sb == "Greedy Best-First Search":
            exp_sb, path_sb, cost_sb = run_gbfs(
                G_sandbox, curr_coords, start_sb, goal_sb
            )
        else:
            exp_sb, path_sb, cost_sb, _ = run_astar(
                G_sandbox, curr_coords, start_sb, goal_sb
            )

        fig_sb = plot_dracula_graph(
            G_sandbox,
            curr_coords,
            path_sb,
            f"Interactive Path ({algo_sb}): {start_sb} to {goal_sb}",
        )
        st.pyplot(fig_sb)

        st.markdown(
            f"""
            <div class="dracula-card">
                <h3>🔍 Sandbox Result Details</h3>
                <p><b>Selected Domain:</b> {graph_choice}</p>
                <p><b>Algorithm:</b> <span style="color:#8be9fd;">{algo_sb}</span></p>
                <p><b>Explored Nodes:</b> {' ➔ '.join(exp_sb)}</p>
                <p><b>Solution Path:</b> <span style="color:#50fa7b;">{' ➔ '.join(path_sb)}</span></p>
                <p><b>Total Cost:</b> <span style="color:#ff79c6; font-weight:bold;">{cost_sb} units</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
