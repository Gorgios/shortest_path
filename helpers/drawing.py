import networkx as nx

import algorithms
import helpers.my_networkx as my_nx
import matplotlib.pyplot as plt
import cv2
import graph_utils as gu

arc_rad = 0.25


def draw_points_on_image(points, image_path, output, size = 1):
    output_image = cv2.imread(f'grids/raw_images/{image_path}')
    for coords in points:
        cv2.circle(output_image, coords, size, (0,0,255), -1)
    cv2.imwrite(f"grids/output_images/{output}", output_image)



def draw_paths_on_image(output, image_file, edges_output, points, diagonal=True, JPS=False):
    edges = f'grids/edges/{edges_output}'
    image = gu.get_edges_from_maze_image(f'grids/raw_images/{image_file}', edges, diagonal=diagonal)
    g = gu.load_graph_from_edges(edges, directed=False, weighted=diagonal, delimiter=":", tuples=True)
    print(g.number_of_nodes())
    print(g.number_of_edges())
    output_image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    count = 0
    for point in points:
        path = nx.bidirectional_shortest_path(g, point[0], point[1]) if not diagonal \
            else algorithms.jps(g, point[0], point[1], heuristic=algorithms.euclides) if JPS else \
            nx.astar_path(g, point[0], point[1], heuristic=algorithms.euclides)
        for coords in path:
            cv2.circle(output_image, coords, 1, colors[count], -1)
        cv2.circle(output_image, path[0], 10, colors[count], thickness=-1)
        cv2.circle(output_image, path[len(path) - 1], 10, colors[count], thickness=-1)
        count += 1
    cv2.imwrite(f"grids/output_images/{output}", output_image)


def draw_road_map(G, coords, filename, paths=None, points=None, color_edges=True):
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(20, 10)
    pos = {}
    for value in coords.values():
        pos[value] = (value[1], value[0])
    nx.draw_networkx(G, pos, ax=ax, node_size=0.2, with_labels=False, font_weight=0.5, width=0.4, node_color="black")
    colors = ['r', 'r', 'b', 'b', 'g', 'g']
    if paths is not None:
        count = 0
        for path in paths:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=colors[count], node_size=0.2)
            if color_edges:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=colors[count], width=0.6)
            count += 2
    if points is not None:
        if len(points) == 6:
            nx.draw_networkx_nodes(G, pos, nodelist=points, node_color=colors, node_size=100)
    plt.savefig(f"road_networks/images/{filename}", dpi=800)


def draw_graph(G, filename, pos=None, directed=False, weighted=False, node_size=30, with_labels=True, font_size=2,
               dpi=1000,
               width=20, height=10, coloured_node=None, edge_width=0.1):
    if pos is None:
        pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(width, height)
    if directed:
        _directed_graph(G, pos, ax, weighted, node_size, with_labels, font_size, edge_width)
    else:
        _undirected_graph(G, pos, ax, weighted, node_size, with_labels, font_size, edge_width)
    if coloured_node is not None:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, nodelist=coloured_node, node_color="red")
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)


def _undirected_graph(G: nx.Graph, pos, ax, weighted, node_size, with_labels, font_size, edge_width):
    nx.draw_networkx(G, pos, ax=ax, node_size=node_size, with_labels=with_labels, font_weight=font_size,
                     width=edge_width)
    if weighted:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def _directed_graph(G: nx.DiGraph, pos, ax, weighted, node_size, with_labels, font_size, edge_width):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight=font_size)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=_get_straight_edges(G), width=edge_width)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=_get_curved_edges(G), connectionstyle=f'arc3, rad = {arc_rad}',
                           width=edge_width)
    if weighted:
        _draw_weights(G, pos, ax, font_size)


def _draw_weights(G: nx.DiGraph, pos, ax, font_size):
    edge_weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: edge_weights[edge] for edge in _get_curved_edges(G)}
    straight_edge_labels = {edge: edge_weights[edge] for edge in _get_straight_edges(G)}
    my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad,
                                       font_size=font_size)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, font_size=font_size)


def _get_curved_edges(G: nx.DiGraph):
    return [edge for edge in G.edges() if reversed(edge) in G.edges()]


def _get_straight_edges(G: nx.DiGraph):
    return list(set(G.edges()) - set(_get_curved_edges(G)))
