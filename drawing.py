import networkx as nx
import helpers.my_networkx as my_nx
import matplotlib.pyplot as plt

arc_rad = 0.25


def draw_graph(G, filename, directed=False, weighted=False, node_size=30, with_labels=True, font_weight=2, dpi=1000,
               width=20, height=10):
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(width, height)
    if directed:
        _directed_graph(G, pos, ax, weighted, node_size, with_labels, font_weight)
    else:
        _undirected_graph(G, pos, ax, weighted, node_size, with_labels, font_weight)
    plt.savefig(filename, dpi=dpi)


def _undirected_graph(G: nx.Graph, pos, ax, weighted, node_size, with_labels, font_weight):
    nx.draw_networkx(G, pos, ax=ax, node_size=node_size, with_labels=with_labels, font_weight=font_weight)
    if weighted:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def _directed_graph(G: nx.DiGraph, pos, ax, weighted, node_size, with_labels, font_weight):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight=font_weight)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=_get_straight_edges(G))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=_get_curved_edges(G), connectionstyle=f'arc3, rad = {arc_rad}')
    if weighted:
        _draw_weights(G, pos, ax, font_weight)


def _draw_weights(G: nx.DiGraph, pos, ax, font_weight):
    edge_weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: edge_weights[edge] for edge in _get_straight_edges(G)}
    straight_edge_labels = {edge: edge_weights[edge] for edge in _get_straight_edges(G)}
    my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad,
                                       font_weight=font_weight)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, font_weight=font_weight)


def _get_curved_edges(G: nx.DiGraph):
    return [edge for edge in G.edges() if reversed(edge) in G.edges()]


def _get_straight_edges(G: nx.DiGraph):
    return list(set(G.edges()) - set(_get_curved_edges(G)))
