# coding=utf8

"""
Build Graph
"""

import copy
import random
from collections import deque
import numpy as np
from pprint import pprint
from func_set import FUNCTIONS, FunctionType, select_function


MAX_CHILDREN = 2
MIN_CHILDREN = 1


class NodeType:
    LEFT_NODE = "LEAF"
    INTERNAL_NODE = "INTERNAL"
    ROOT_NODE = "ROOT"


class Node:

    curr_id = 0

    def __init__(self, node_type):
        self.node_type = node_type
        self.children = list()
        self.parents = list()
        self.level = -1
        self._id = Node.curr_id
        Node.curr_id += 1

    @property
    def id(self):
        return self._id

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parents.append(parent)

    def erase_states(self):
        self.children = []
        self.parents = []
        self.level = -1


class Graph:

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.vertices_attrs = {}

    def indegree(self, vertex_label):
        count = 0
        for edge in self.edges:
            if edge[1] == vertex_label:
                count += 1
        return count

    def outdegree(self, vertex_label):
        count = 0
        for edge in self.edges:
            if edge[0] == vertex_label:
                count += 1
        return count

    def parents(self, vertex_label):
        vertices = list()
        for edge in self.edges:
            if edge[1] == vertex_label:
                vertices.append(edge[0])
        return vertices

    def children(self, vertex_label):
        vertices = list()
        for edge in self.edges:
            if edge[0] == vertex_label:
                vertices.append(edge[1])
        return vertices

    def set_attr(self, vertex_label, attrs):
        if vertex_label not in self.vertices_attrs:
            self.vertices_attrs[vertex_label] = {}
        self.vertices_attrs[vertex_label].update(attrs)

    def get_attr(self, vertex_label, attr_name):
        if vertex_label not in self.vertices_attrs:
            return None
        if attr_name not in self.vertices_attrs[vertex_label]:
            return None
        return self.vertices_attrs[vertex_label][attr_name]

    def print(self):
        print(self.vertices)
        print(self.edges)
        pprint(self.vertices_attrs)


def topological_sort(graph):

    UNMARKED = "UNMMARKED"
    TEMPORAL_MARK = "TEMPORAL_MARK"
    PERMANENT_MARK = "PERMANENT_MARK"

    sorted_vertices = list()
    vertices = copy.deepcopy(graph.vertices)

    for v in vertices:
        graph.set_attr(v, {
            "mark": UNMARKED
        })

    def _visit(_v):
        _m = graph.get_attr(_v, "mark")
        assert _m != TEMPORAL_MARK
        if _m == UNMARKED:
            graph.set_attr(_v, {
                "mark": TEMPORAL_MARK
            })
            children = graph.children(_v)
            for child in children:
                _visit(child)
            graph.set_attr(_v, {
                "mark": PERMANENT_MARK
            })
            sorted_vertices.append(_v)

    is_done = False
    curr_idx = 0
    while not is_done:
        next_vertex = None

        while curr_idx < len(vertices):
            v = vertices[curr_idx]
            m = graph.get_attr(v, "mark")
            curr_idx += 1
            if m == UNMARKED:
                next_vertex = v
                break
        else:
            is_done = True

        _visit(next_vertex)

    reversed_vertices = list(reversed(sorted_vertices))

    print("Topological Sort: ", reversed_vertices)
    return reversed_vertices


def dfs_to_string(root_node, string=""):
    string += "( %d " % root_node.id
    for node in root_node.children:
        string = dfs_to_string(node, string)
    string += ") %d " % root_node.id
    return string


def bfs_to_label_level(root_node):
    node_stack = deque()
    node_stack.append(root_node)
    root_node.level = 0
    count = 1
    next_count = 0
    curr = 0
    level = 0
    while len(node_stack):
        node = node_stack.popleft()
        curr += 1
        node.level = level
        for child_node in node.children:
            node_stack.append(child_node)
        next_count += len(node.children)

        if curr == count:
            # print("next")
            level += 1
            count = next_count
            curr = 0
            next_count = 0


def erase_node_states(nodes):
    for node in nodes:
        node.erase_states()


def generate(num_input_node, num_internal_node):
    """
    Generate random data flow graph
    :param num_input_node:
    :param num_internal_node:
    :return:
        Data flow graph
    """
    root_node = Node(NodeType.ROOT_NODE)
    nodes = list()
    for i in range(num_input_node + num_internal_node):
        nodes.append(Node(NodeType.INTERNAL_NODE))

    highest_level = -1
    is_valid = False
    while not is_valid:

        node_stack = deque()
        node_stack.append(root_node)
        # only copy the list, but not the object
        candidates = copy.copy(nodes)

        while len(node_stack) > 0 and len(candidates) > 0:

            node = node_stack.popleft()

            if len(candidates) < MAX_CHILDREN:
                num_children = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                # 4 functions require 2 arguments, 11 functions require only 1 arguments
                num_children = np.random.choice([0, 1, 2], p=[1/3, 22/45, 8/45])

            # num_children = random.randint(0, min(MAX_CHILDREN, len(candidates)))

            selected_nodes = random.sample(candidates, num_children)

            for n in selected_nodes:
                candidates.remove(n)
                node.add_child(n)
                n.set_parent(node)
                node_stack.append(n)

        if len(candidates) == 0:
            actual_leaves = [n for n in nodes if len(n.children) == 0]
            # Ensure enough leaf nodes
            if len(actual_leaves) >= num_input_node:
                bfs_to_label_level(root_node)
                highest_level = max([n.level for n in nodes])
                highest_level_node_count = 0
                for n in nodes:
                    if n.level == highest_level:
                        highest_level_node_count += 1
                # Ensure not too much leaf nodes
                if highest_level_node_count <= num_input_node:
                    is_valid = True
                else:
                    erase_node_states(nodes + [root_node])
            else:
                erase_node_states(nodes + [root_node])
        else:
            erase_node_states(nodes + [root_node])

    string = dfs_to_string(root_node)
    print("tree: %s " % string)

    # Build Graph
    edges = list()
    vertices = list()
    for n in (nodes + [root_node]):
        vertices.append(n.id)
        for child in n.children:
            edges.append((child.id, n.id))

    print(edges)
    print(vertices)

    # Select Leaf
    leaves = [n for n in nodes if n.level == highest_level]
    left = num_input_node - len(leaves)
    actual_leaves = [n for n in nodes if len(n.children) == 0 and n.level != highest_level]
    extra_leaves = random.sample(actual_leaves, left)
    for leaf_node in extra_leaves:
        leaf_node.node_type = NodeType.LEFT_NODE
        actual_leaves.remove(leaf_node)

    # Eliminate unnecessary leaves
    actual_leaves.sort(key=lambda n: n.level)
    for al in actual_leaves:
        candidates = [n for n in nodes if n.level >= al.level and n != al]
        selected_node = random.sample(candidates, 1)[0]
        edges.append((selected_node.id, al.id))
    print("==================================")
    print("Final Graph: ")
    print(vertices)
    print(edges)

    return Graph(vertices, edges)


def assign_variable_name(graph):
    """
    Assign variable name to vertex
    :param graph:
    :return:
    """
    variable_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "x", "y"]

    for idx, vertex in enumerate(graph.vertices):
        graph.set_attr(vertex, {
            "variable_name": variable_names[idx]
        })


def add_function(graph):
    """
    Add function to edge and function name
    :param graph:
    :return:
    """

    node_stack = deque()
    root_node = 0
    node_stack.append(root_node)

    while len(node_stack):
        node = node_stack.popleft()
        parent = graph.parents(node)

        if len(parent) == 0:
            continue

        funcs = select_function(
            num_arguments=len(parent),
            return_type=graph.get_attr(node, "data_type")
        )

        assert len(funcs) > 0

        func = random.sample(funcs, 1)[0]
        func_definition = FUNCTIONS[func]

        graph.set_attr(node, {
            "data_type": func_definition["return_type"],
            "func": func
        })

        if func_definition["func_type"] == FunctionType.FIRST_ORDER:
            arguments = func_definition["arguments"]
        else:
            arguments = func_definition["arguments"][1:]

        for (n, data_type) in zip(parent, arguments):
            n_type = graph.get_attr(n, "data_type")
            if not n_type:
                graph.set_attr(n, {
                    "data_type": data_type
                })
                node_stack.append(n)
            else:
                if n_type != data_type:
                    raise Exception("Data Type mismatch")

    graph.print()


def generate_program(graph, sorted_vertices):
    for v in sorted_vertices:
        func = graph.get_attr(v, "func")
        variable_name = graph.get_attr(v, "variable_name")
        if not func:
            # Input node
            print(variable_name)
        else:
            print(variable_name, " = ", func, '(', ', '.join([graph.get_attr(v, "variable_name") for v in graph.parents(v)]), ")")


if __name__ == "__main__":
    data_flow_graph = generate(1, 5)
    assign_variable_name(data_flow_graph)
    add_function(data_flow_graph)
    sort_result = topological_sort(data_flow_graph)
    generate_program(data_flow_graph, sort_result)
