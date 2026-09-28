import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# ==========================================
# 1. PROBLEM DOMAIN DATASETS
# ==========================================

# Task 1 Dataset: Warehouse Robot
warehouse_locations = {
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

# Task 2 & 3 Dataset: Airport / Emergency Hospital
hospital_locations = {
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

# Task 4 Dataset: Autonomous Drone
drone_locations = {
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


# Helper function to construct graph
def build_graph(locations_dict, edges_list):
    G = nx.Graph()
    for node, pos in locations_dict.items():
        G.add_node(node, pos=pos)
    for u, v, w in edges_list:
        G.add_edge(u, v, weight=w)
    return G


def heuristic(n1, n2, locations):
    x1, y1 = locations[n1]
    x2, y2 = locations[n2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ==========================================
# 2. SEARCH ALGORITHMS
# ==========================================


def run_gbfs(graph, locations, start, goal):
    pq = [(heuristic(start, goal, locations), start)]
    came_from = {start: None}
    visited = set()
    expansion_sequence = []

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        expansion_sequence.append(current)

        if current == goal:
            break

        for nxt in graph.neighbors(current):
            if nxt not in visited and nxt not in came_from:
                came_from[nxt] = current
                heapq.heappush(pq, (heuristic(nxt, goal, locations), nxt))

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    cost = sum(
        graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
    )
    return expansion_sequence, path, round(cost, 2)


def run_astar(graph, locations, start, goal, weight=1.0):
    pq = [(weight * heuristic(start, goal, locations), start)]
    g_score = {node: float("inf") for node in graph.nodes()}
    g_score[start] = 0
    came_from = {start: None}
    visited = set()
    expansion_sequence = []
    metrics = []

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        expansion_sequence.append(current)

        g_val = g_score[current]
        h_val = heuristic(current, goal, locations)
        metrics.append(
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
                f_score = tentative_g + weight * heuristic(
                    nxt, goal, locations
                )
                heapq.heappush(pq, (f_score, nxt))

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    return expansion_sequence, path, round(g_score[goal], 2), metrics


# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================

st.title("AI Informed Search Algorithm Visualizer")
st.write(
    "Explore Greedy Best-First Search (GBFS) and A* Search across real-world problem scenarios."
)

st.sidebar.header("Search Configurations")

# Scenario Selector
scenario_choice = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Hospital Emergency Supply (Task 3)",
        "Airport Baggage Handling (Task 2)",
        "Warehouse Robot Navigation (Task 1)",
        "Autonomous Drone Delivery (Task 4)",
    ],
)

# Load selected graph data
if "Warehouse" in scenario_choice:
    curr_locations, curr_edges = warehouse_locations, warehouse_edges
elif "Drone" in scenario_choice:
    curr_locations, curr_edges = drone_locations, drone_edges
else:
    curr_locations, curr_edges = hospital_locations, hospital_edges

G = build_graph(curr_locations, curr_edges)
nodes_list = list(curr_locations.keys())

start_node = st.sidebar.selectbox("Select Start Node", nodes_list, index=0)
goal_node = st.sidebar.selectbox(
    "Select Goal Node", nodes_list, index=len(nodes_list) - 1
)

algo_choice = st.sidebar.selectbox(
    "Select Search Algorithm",
    [
        "Greedy Best-First Search (GBFS)",
        "A* Search",
        "Weighted A* Search",
    ],
)

# Slider for Weighted A*
weight_val = 1.0
if algo_choice == "Weighted A* Search":
    weight_val = st.sidebar.slider(
        "Heuristic Weight (w)", 1.0, 3.0, 1.5, step=0.5
    )

if st.sidebar.button("Run Search"):
    if start_node == goal_node:
        st.warning("Start and Goal nodes must be different!")
    else:
        # Execute Algorithm
        if algo_choice == "Greedy Best-First Search (GBFS)":
            exp_seq, path, cost = run_gbfs(
                G, curr_locations, start_node, goal_node
            )
            metrics = []
        elif algo_choice == "A* Search":
            exp_seq, path, cost, metrics = run_astar(
                G, curr_locations, start_node, goal_node, weight=1.0
            )
        else:
            exp_seq, path, cost, metrics = run_astar(
                G, curr_locations, start_node, goal_node, weight=weight_val
            )

        # Plot Network Graph
        fig, ax = plt.subplots(figsize=(9, 5.5))
        pos = nx.get_node_attributes(G, "pos")
        path_edges = list(zip(path[:-1], path[1:]))

        # Node highlights
        unvisited = [n for n in G.nodes() if n not in path]
        intermediates = [n for n in path if n not in (start_node, goal_node)]

        # Draw default nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=unvisited,
            ax=ax,
            node_size=600,
            node_color="#E0E0E0",
            edgecolors="#888888",
        )
        # Draw path intermediate nodes
        if intermediates:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=intermediates,
                ax=ax,
                node_size=650,
                node_color="#FFB74D",
                edgecolors="#E65100",
            )
        # Draw start node
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[start_node],
            ax=ax,
            node_size=750,
            node_color="#64B5F6",
            edgecolors="#0D47A1",
        )
        # Draw goal node
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[goal_node],
            ax=ax,
            node_size=750,
            node_color="#E57373",
            edgecolors="#B71C1C",
        )

        # Draw graph edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=1.5, edge_color="#CCCCCC", alpha=0.8
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, ax=ax, width=3.5, edge_color="#2E7D32"
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=9, font_weight="bold"
        )

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        plt.title(f"Path Visualizer: {algo_choice}", fontsize=12)
        plt.axis("off")
        st.pyplot(fig)

        # Deliverables Output Section
        st.subheader("Execution Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Algorithm", algo_choice)
        col2.metric("Total Path Cost", f"{cost} units")
        col3.metric("Nodes Expanded", len(exp_seq))

        st.write(f"**Node Expansion Order:** `{' ➔ '.join(exp_seq)}`")
        st.write(f"**Solution Path:** `{' ➔ '.join(path)}`")

        # Display Metrics Table if available
        if metrics:
            st.subheader("Expanded Nodes Step-by-Step Metrics")
            st.table(metrics)
