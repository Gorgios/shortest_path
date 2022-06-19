import math
import random

import cv2
import networkx as nx
import helpers.drawing as draw
import algorithms as algo

d_val = math.sqrt(2)


def generate_and_save_graph(nodes, edges, directed=False, weighted=False, node_size=50, edge_width=0.2, threshold=0.5,
                            image=True, font_size=1):
    G = generate_random_graph(nodes, edges, directed=directed, weighted=weighted)
    node_to_analyze = get_node_to_analyze(G, threshold)
    file_name = f'random_{"weighted" if weighted else "unweighted"}_{"directed" if directed else "undirected"}_{nodes}_{edges} '
    with open(f'analyzed_nodes/{file_name}.txt', "w+") as f:
        f.write(str(node_to_analyze))
    if image:
        draw.draw_graph(G, f'graph_images/{file_name}.png',
                        directed=directed, weighted=weighted, node_size=node_size,
                        with_labels=False, dpi=1000, edge_width=edge_width, coloured_node=[node_to_analyze],
                        font_size=font_size)
    save_graph_as_edge_list(G, f'graph_edges/{file_name}.txt')


def get_road_network_graph_and_coords(nodes_file, edges_file):
    with open(f'road_networks/data/{nodes_file}', encoding="utf8") as f:
        data = f.readlines()
    coords = dict()
    for i in data:
        split = i.split(" ")
        coords[int(split[0])] = float(split[2]), float(split[1])
    with open(f'road_networks/data/{edges_file}', encoding="utf8") as f:
        data = f.readlines()
    edges = list()
    for i in data:
        split = i.split(" ")
        c1 = coords[int(split[1])]
        c2 = coords[int(split[2])]
        edges.append((c1, c2, algo.euclides(c1, c2)))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G, coords


def get_road_network_points(node_file, edges_file):
    G, coords = get_road_network_graph_and_coords(node_file, edges_file)
    sorted_coords = list(coords.values())
    sorted_coords.sort(key=lambda x: x[0])
    points = []
    for i in range(0, len(sorted_coords)):
        if nx.has_path(G, sorted_coords[i], sorted_coords[len(sorted_coords) - 1]):
            points.append((sorted_coords[0], sorted_coords[len(sorted_coords) - 1]))
            break
    sorted_coords.sort(key=lambda y: y[1])
    for i in range(10, len(sorted_coords)):
        if nx.has_path(G, sorted_coords[i], sorted_coords[len(sorted_coords) - 11]):
            points.append((sorted_coords[i], sorted_coords[len(sorted_coords) - 11]))
            break
    for i in range(int(len(sorted_coords)/ 3), int(len(sorted_coords)/3 * 2)):
        if nx.has_path(G, sorted_coords[i], sorted_coords[len(sorted_coords) - int(len(sorted_coords)/3 * 2)]):
            points.append((sorted_coords[i], sorted_coords[int(len(sorted_coords) / 3 * 2)]))
            break
    print(points)
    return points


def generate_random_graph(nodes, edges, directed=False, weighted=False, weights_from=1, weights_to=10):
    graph = nx.gnm_random_graph(nodes, edges, directed=directed)
    if weighted:
        for e in graph.edges():
            graph[e[0]][e[1]]['weight'] = random.randint(weights_from, weights_to)

    return graph


def save_graph_as_edge_list(graph, filename):
    edges = graph.edges()
    with open(filename, "w+") as f:
        for e in edges:
            f.write(f'{e[0]},{e[1]},{graph[e[0]][e[1]]["weight"]}\n')


def get_node_to_analyze(graph, threshold=0.5):
    for i in graph.nodes:
        if len(list(nx.dfs_preorder_nodes(graph, i))) / len(graph.nodes) > threshold:
            return i
    raise Exception("There is not good candidate for given threshold")


def get_node_to_analyze_from_file(filename):
    with open(filename) as f:
        return int(f.read())


def load_graph_from_edges(filename, directed=True, weighted=True, delimiter=" ", tuples=False):
    edges = list()
    with open(filename) as f:
        lines = f.readlines()
    for i in lines:
        str = i.strip('\n')
        split = str.split(delimiter)
        if tuples:
            split_tuple_1 = split[0].strip('(').strip(')').split(",")
            split_tuple_2 = split[1].strip('(').strip(')').split(",")

            u = (int(split_tuple_1[0]), int(split_tuple_1[1]))
            v = (int(split_tuple_2[0]), int(split_tuple_2[1]))
        else:
            u = int(split[0])
            v = int(split[1])
        if weighted:
            edges.append((u, v, float(split[2])))
        else:
            edges.append((u, v))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_weighted_edges_from(edges) if weighted else G.add_edges_from(edges)

    return G


def get_edges_from_maze_image(image_file, output_file, diagonal=True):
    img = cv2.imread(image_file)
    gray_image = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    col, rows = blackAndWhiteImage.shape[1], blackAndWhiteImage.shape[0]
    edges = list()
    G = nx.DiGraph()
    for i in range(0, rows):
        for j in range(0, col):
            if _check_if_white(blackAndWhiteImage, i, j):
                if diagonal:
                    edges += _add_edges_diagonal(blackAndWhiteImage, i, j)
                else:
                    edges += _add_edges(blackAndWhiteImage, i, j)
    with open(output_file, 'w') as fp:
        for i in edges:
            fp.write(("{}:{}:{}\n".format(i[0], i[1], i[2])))
    return blackAndWhiteImage


def _check_if_white(image, i, j):
    return len(image[1]) > i >= 0 and len(image) > j >= 0 and image[j][i][0] == 255


def _add_edges(image, i, j):
    edges = list()
    if _check_if_white(image, i, j + 1):
        edges.append(((i, j), (i, j + 1), 1.0))
    if _check_if_white(image, i, j - 1):
        edges.append(((i, j), (i, j - 1), 1.0))
    if _check_if_white(image, i - 1, j):
        edges.append(((i, j), (i - 1, j), 1.0))
    if _check_if_white(image, i + 1, j):
        edges.append(((i, j), (i + 1, j), 1.0))
    return edges


def _add_edges_diagonal(image, i, j):
    edges = _add_edges(image, i, j)
    if _check_if_white(image, i + 1, j + 1):
        edges.append(((i, j), (i + 1, j + 1), d_val))
    if _check_if_white(image, i - 1, j - 1):
        edges.append(((i, j), (i - 1, j - 1), d_val))
    if _check_if_white(image, i + 1, j - 1):
        edges.append(((i, j), (i + 1, j - 1), d_val))
    if _check_if_white(image, i - 1, j + 1):
        edges.append(((i, j), (i - 1, j + 1), d_val))
    return edges
