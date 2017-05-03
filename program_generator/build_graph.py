# coding=utf8

"""
Build Graph
"""

import copy
import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt


MAX_CHILDREN = 3
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


def visualize_graph(vertices, edges):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    nx.draw(G)
    plt.show()


def generate(num_input_node, num_internal_node):

    root_node = Node(NodeType.ROOT_NODE)
    nodes = list()
    for i in range(num_input_node + num_internal_node):
        nodes.append(Node(NodeType.INTERNAL_NODE))

    highest_level = -1
    is_valid = False
    while not is_valid:

        node_stack = [root_node]
        # only copy the list, but not the object
        candidates = copy.copy(nodes)

        while len(node_stack) > 0 and len(candidates) > 0:

            node = node_stack.pop()

            num_children = random.randint(0, min(MAX_CHILDREN, len(candidates)))

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
    visualize_graph(vertices, edges)

if __name__ == "__main__":
    generate(2, 7)
