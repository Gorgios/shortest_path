import networkx as nx
import pandas as pd
import os
import graph_utils as gu
import algorithms as alg
import algo_stats as alg_stats
from helpers.timer import Timer


def get_visited_nodes_grids(output, edges_output, image, points, which_to_analyze = 1, iterations=10):
    gu.get_edges_from_maze_image(image, edges_output, diagonal=True)
    gu.get_edges_from_maze_image(image, f'non-diag-{edges_output}', diagonal=False)
    g_diag = gu.load_graph_from_edges(edges_output, directed=False, weighted=True, delimiter=":", tuples=True)
    g_no_diag = gu.load_graph_from_edges(f'non-diag-{edges_output}', directed=False, weighted=False, delimiter=":", tuples=True)
    a_star_diag = []
    a_star_euc = []
    a_star_man = []
    bi_dir = []
    jps = []
    count = 0
    res = None, None, None, None, None
    for point in points:
        path, diag_a_enqueued = alg_stats.astar_visited_nodes(g_diag, point[0], point[1], heuristic=alg.euclides)
        a_star_diag.append(len(diag_a_enqueued))
        path, cost, jps_enqueued = alg.jps(g_diag, point[0], point[1], heuristic=alg.euclides)
        jps.append(len(jps_enqueued))
        path, eu_enqueued = alg_stats.astar_visited_nodes(g_no_diag, point[0], point[1], heuristic=alg.euclides)
        a_star_euc.append(len(eu_enqueued))
        path, ma_enqueued = alg_stats.astar_visited_nodes(g_no_diag, point[0], point[1], heuristic=alg.manhattan)
        print(nx.astar_path_length(g_no_diag, point[0], point[1], heuristic=alg.manhattan))
        print(nx.astar_path_length(g_no_diag, point[0], point[1], heuristic=alg.euclides))
        a_star_man.append(len(ma_enqueued))
        path, succ, pred= alg_stats.bidirectional_shortest_path_visited_nodes(g_no_diag, point[0], point[1])
        bi_dir.append(len(succ) + len(pred))
        succ.update(pred)
        if count == which_to_analyze:
            res = diag_a_enqueued, jps_enqueued, eu_enqueued, ma_enqueued, succ
        count += 1
    df_diag = pd.DataFrame(data={"Para": points, "JPS": jps, "A-str": a_star_diag})
    df_no_diag = pd.DataFrame(data={"Para": points,"A-star euclides": a_star_euc, "A-star manhattan": a_star_man, "Bidirectional BFS": bi_dir})
    df_diag.to_csv(f'results/diag-{output}', index=False)
    df_no_diag.to_csv(f'results/no-diag-{output}', index=False)
    return res


def test_grids(output, edges_output, image, points, iterations=10):
    t = Timer()
    gu.get_edges_from_maze_image(f'/grids/raw_images/{image}', f'/grids/edges/{edges_output}', diagonal=True)
    gu.get_edges_from_maze_image(f'/grids/raw_images/{image}', f'/grids/edges/non-diag-{edges_output}', diagonal=False)
    g_diag = gu.load_graph_from_edges(f'/grids/edges/{edges_output}', directed=False, weighted=True, delimiter=":", tuples=True)
    g_no_diag = gu.load_graph_from_edges(f'/grids/edges/non-diag-{edges_output}', directed=False, weighted=False, delimiter=":", tuples=True)
    a_star_diag = []
    a_star_euc = []
    a_star_man = []
    bi_dir = []
    jps = []
    for point in points:
        t.start()
        for i in range(0, iterations):
            nx.astar_path(g_diag, point[0], point[1], heuristic=alg.euclides)
        a_star_diag.append(t.stop() / iterations)
        t.start()
        for i in range(0, iterations):
            alg.jps(g_diag, point[0], point[1], heuristic=alg.euclides)
        jps.append(t.stop() / iterations)
        t.start()
        for i in range(0, iterations):
            nx.astar_path(g_no_diag, point[0], point[1], heuristic=alg.euclides)
        a_star_euc.append(t.stop() / iterations)
        t.start()
        for i in range(0, iterations):
            nx.astar_path(g_no_diag, point[0], point[1], heuristic=alg.manhattan)
        a_star_man.append(t.stop() / iterations)
        t.start()
        for i in range(0, iterations):
            nx.bidirectional_shortest_path(g_no_diag, point[0], point[1])
        bi_dir.append(t.stop() / iterations)
    df_diag = pd.DataFrame(data={"Para": points, "JPS": jps, "A-str": a_star_diag, "Iterations": [iterations] * len(points)})
    df_no_diag = pd.DataFrame(data={"Para": points,"A-star euclides": a_star_euc, "A-star manhattan": a_star_man, "Bidirectional BFS": bi_dir, "Iterations": [iterations] * len(points)})
    df_diag.to_csv(f'results/diag-{output}', index=False)
    df_no_diag.to_csv(f'results/no-diag-{output}', index=False)


def test_spt(graph_data, output, iterations=10, tuples=False, delimiter=",", directed=False, weighted=False,
             res_format="{:0.6f}"):
    name, file = graph_data
    node = gu.get_node_to_analyze_from_file(f'analyzed_nodes/{file}')
    G = gu.load_graph_from_edges(f'graph_edges/{file}', directed, weighted, delimiter, tuples)
    t = Timer()
    bfs = 0
    if not weighted:
        t.start()
        for i in range(0, iterations):
            nx.predecessor(G, node)
        bfs = res_format.format(t.stop() / iterations)
    t.start()
    for i in range(0, iterations):
        nx.dijkstra_predecessor_and_distance(G, node)
    dijkstra = res_format.format(t.stop() / iterations)
    t.start()
    for i in range(0, iterations):
        nx.bellman_ford_predecessor_and_distance(G, node)
    bellman = res_format.format(t.stop() / iterations)
    df = pd.DataFrame(
        data={'Graf': [name], 'BFS': [bfs], 'Bellman': [bellman], 'Dijkstra': [dijkstra], 'Iteracje': [iterations]})
    if weighted:
        df.drop(columns=["BFS"])
    df.to_csv(output, mode='a', header=not os.path.exists(output), index=False)
    return df


def test_road_network(output, nodefile, edgefile, points, iterations=5, res_format="{:0.6f}"):
    G, coords = gu.get_road_network_graph_and_coords(nodefile, edgefile)
    bellman = []
    a_star = []
    dijkstra = []
    bi_dijkstra = []
    t = Timer()
    for point in points:
        t.start()
        for i in range(0, iterations):
            nx.dijkstra_path(G, point[0], point[1])
        dijkstra.append(res_format.format(t.stop() / iterations))
        t.start()
        for i in range(0, iterations):
            nx.bellman_ford_path(G, point[0], point[1])
        bellman.append(res_format.format(t.stop() / iterations))
        t.start()
        for i in range(0, iterations):
            nx.astar_path(G, point[0], point[1], heuristic=alg.euclides)
        a_star.append(res_format.format(t.stop() / iterations))
        t.start()
        for i in range(0, iterations):
            nx.bidirectional_dijkstra(G, point[0], point[1])
        bi_dijkstra.append(res_format.format(t.stop() / iterations))
    df = pd.DataFrame({'Analizowana para': points, 'A-star': a_star, 'Bellman': bellman, 'Dijkstra': dijkstra,
                       "Bi-dijkstra": bi_dijkstra, 'Iteracje': [iterations] * len(points)})
    df.to_csv(f'results/{output}', index=False)


def get_visited_nodes_for_road_network(output, nodefile, edgefile, points):
    G, coords = gu.get_road_network_graph_and_coords(nodefile, edgefile)
    a_star = []
    dijkstra = []
    bi_dijkstra = []
    a_star_enq = []
    dijkstra_enq = []
    bi_dijkstra_enq = []
    for point in points:
        path, enqueued = alg_stats.astar_visited_nodes(G, point[0], point[1], heuristic=alg.euclides)
        a_star.append(len(enqueued))
        a_star_enq.append(enqueued)
        path, enqueued = alg_stats.dijkstra_visited_nodes(G, point[0], point[1])
        dijkstra.append(len(enqueued))
        dijkstra_enq.append(enqueued)
        path, enqueued = alg_stats.bidirectional_dijkstra_visited_nodes(G, point[0], point[1])
        bi_dijkstra.append(len(enqueued[0]) + len(enqueued[1]))
        bi_dijkstra_enq.append(enqueued)
    df = pd.DataFrame({'Analizowana para': points, 'A-star': a_star, 'Dijkstra': dijkstra, "Bi-dijkstra": bi_dijkstra})
    df.to_csv(f'results/{output}', index=False)
    return a_star_enq, dijkstra_enq, bi_dijkstra_enq
