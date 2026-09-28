import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# 1. Define Common Graph Data
locations = {
    "Pharmacy": (0, 0),
    "Corridor_A": (1, 4),
    "Corridor_B": (2, 1),
    "ICU": (5, 5),
    "Triage": (4, 2),
    "Radiology": (7, 4),
    "Emergency_Ward": (8, 6),
}

edges = [
    ("Pharmacy", "Corridor_A", 4.12),
    ("Pharmacy", "Corridor_B", 2.24),
    ("Corridor_A", "ICU", 5.00),
    ("Corridor_B", "Triage", 2.24),
    ("ICU", "Emergency_Ward", 3.16),
    ("Triage", "Radiology", 3.16),
    ("Radiology", "Emergency_Ward", 2.24),
]

G = nx.Graph()
for node, pos in locations.items():
    G.add_node(node, pos=pos)
for u, v, w in edges:
    G.add_edge(u, v, weight=w)


def heuristic(n1, n2):
    x1, y1 = locations[n1]
    x2, y2 = locations[n2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 2. Search Algorithms
def run_gbfs(graph, start, goal):
    pq = [(heuristic(start, goal), start)]
    came_from = {start: None}
    visited = set()
    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        for nxt in graph.neighbors(current):
            if nxt not in visited:
                if nxt not in came_from:
                    came_from[nxt] = current
                heapq.heappush(pq, (heuristic(nxt, goal), nxt))

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    cost = sum(
        graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
    )
    return path, round(cost, 2)


def run_astar(graph, start, goal):
    pq = [(heuristic(start, goal), start)]
    g_score = {node: float("inf") for node in graph.nodes()}
    g_score[start] = 0
    came_from = {start: None}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        for nxt in graph.neighbors(current):
            tentative_g = g_score[current] + graph[current][nxt]["weight"]
            if tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(pq, (f_score, nxt))

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path, round(g_score[goal], 2)


# 3. Streamlit Interface
st.title("AI Informed Search Algorithm Visualizer")
st.sidebar.header("Search Configurations")

nodes_list = list(locations.keys())
start_node = st.sidebar.selectbox("Select Start Node", nodes_list, index=0)
goal_node = st.sidebar.selectbox(
    "Select Goal Node", nodes_list, index=len(nodes_list) - 1
)
algo_choice = st.sidebar.selectbox(
    "Select Search Algorithm", ["Greedy Best-First Search (GBFS)", "A* Search"]
)

if st.sidebar.button("Run Search"):
    if start_node == goal_node:
        st.warning("Start and Goal nodes must be different!")
    else:
        if algo_choice == "Greedy Best-First Search (GBFS)":
            path, cost = run_gbfs(G, start_node, goal_node)
        else:
            path, cost = run_astar(G, start_node, goal_node)

        # Plot Network Graph
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = nx.get_node_attributes(G, "pos")
        path_edges = list(zip(path[:-1], path[1:]))

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=600, node_color="lightblue"
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=path, ax=ax, node_size=700, node_color="coral"
        )
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color="gray")
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, ax=ax, width=3, edge_color="red"
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=9, font_weight="bold"
        )

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        plt.title(f" Path Visualizer: {algo_choice}")
        plt.axis("off")
        st.pyplot(fig)

        # Deliverables Output Section
        st.subheader("Execution Results")
        st.write(f"**Selected Algorithm:** {algo_choice}")
        st.write(f"**Solution Path:** {' ➔ '.join(path)}")
        st.write(f"**Total Path Cost:** {cost} units")