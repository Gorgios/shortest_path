from heapq import heappush, heappop
from itertools import count
import math
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

dir_dict = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3, (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}

ddr = {0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1), 4: (-1, 0), 5: (-1, 1), 6: (0, 1), 7: (1, 1)}

d_val = math.sqrt(2)


def euclides(a1, a2):
    return math.sqrt(math.pow((a1[0] - a2[0]), 2) + math.pow((a1[1] - a2[1]), 2))


def manhattan(a1, a2):
    return math.fabs(a1[0] - a2[0]) + math.fabs(a1[1] - a2[1])


def jps(G, source, target, heuristic=euclides):
    push = heappush
    pop = heappop
    c = count()
    queue = [(0, next(c), source, 0, None, -1)]

    enqueued = {}
    explored = {}
    while queue:
        _, __, curnode, dist, parent, node_dir = pop(queue)
        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, enqueued[target][0], enqueued

        if curnode in explored:
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue
        explored[curnode] = parent
        neighbours = _prune(G, curnode, node_dir)
        for direction in neighbours[0]:
            jpoint, cost = _jump(G, curnode, direction, dist, target)
            if jpoint is None or jpoint in explored:
                continue
            if jpoint in enqueued:
                qcost, h = enqueued[jpoint]
                if qcost <= cost:
                    continue
            else:
                h = heuristic(jpoint, target)
            enqueued[jpoint] = cost, h
            push(queue, (cost + h, next(c), jpoint, cost, curnode, direction))
    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def _calc_dir_of_edge(p1, p2):
    return dir_dict[(p2[0] - p1[0], p2[1] - p1[1])]


def _change_dir(direction, amount):
    return (direction + amount) % 8


def _calc_neighbour(node, dir):
    return node[0] + dir[0], node[1] + dir[1]


def _prune(G: nx.Graph, curnode, direction):
    neighbours = {}
    forced_neighbours_exist = False
    if direction == -1:
        for i in G.neighbors(curnode):
            neighbours[_calc_dir_of_edge(curnode, i)] = i
        return neighbours, False
    is_diagonal = direction % 2
    left = _change_dir(direction, 2 + is_diagonal)
    right = _change_dir(direction, -2 - is_diagonal)
    left_d = _change_dir(left, -1)
    right_d = _change_dir(right, 1)
    for i in range(-is_diagonal, is_diagonal + 1):
        if G.has_node(_calc_neighbour(curnode, ddr[_change_dir(direction, i)])):
            neighbours[_change_dir(direction, i)] = _calc_neighbour(curnode, ddr[_change_dir(direction, i)])
    if len(G.adj[curnode]) == 8:
        return neighbours, False
    if not G.has_node(_calc_neighbour(curnode, ddr[left])) and G.has_node(_calc_neighbour(curnode, ddr[left_d])):
        forced_neighbours_exist = True
        neighbours[left_d] = _calc_neighbour(curnode, ddr[left_d])
    if not G.has_node(_calc_neighbour(curnode, ddr[right])) and G.has_node(_calc_neighbour(curnode, ddr[right_d])):
        forced_neighbours_exist = True
        neighbours[right_d] = _calc_neighbour(curnode, ddr[right_d])

    return neighbours, forced_neighbours_exist


def _step(G: nx.Graph, vertex, direction, cost_so_far):
    next_vertex = _calc_neighbour(vertex, ddr[direction])
    if not G.has_node(next_vertex):
        return None, 0
    cost = cost_so_far + (d_val if direction % 2 == 1 else 1)
    return next_vertex, cost


def _jump(G, vertex, direction, cost_so_far, goal_vertex):
    jump_point, cost = _step(G, vertex, direction, cost_so_far)
    if jump_point is None:
        return None, None
    if jump_point == goal_vertex:
        return jump_point, cost
    if _prune(G, jump_point, direction)[1]:
        return jump_point, cost
    if direction % 2:
        for i in (_change_dir(direction, 1), _change_dir(direction, -1)):
            next_jump_point, _ = _jump(G, jump_point, i, 0, goal_vertex)
            if next_jump_point is not None:
                return jump_point, cost
    jump_point, cost = _jump(G, jump_point, direction, cost, goal_vertex)
    return jump_point, cost
