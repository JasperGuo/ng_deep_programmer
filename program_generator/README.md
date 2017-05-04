## Random Program Generator

#### Basic Idea:

Randomly build a data flow graph, and then assign input nodes data types, and based on the data types and the number of input edges select functions.

#### Node Types

| Node Type     | # input edges | # output edges |
| ------------- | ------------- | -------------- |
| Input Node    | 0             | NOT LIMIT      |
| Output Node   | {1, 2}        | 0              |
| Internal Node | {1, 2}        | NOT LIMIT      |

#### Constrain

1. **DAG**
2. No edges between input nodes
3. Edge Must from lower level to higher level

#### Construction

1. Randomly generate a tree
   1. The number of leaf node > The number of input node
   2. The number of nodes in the highest level <= The number of input node
2. Remove unnecessary leaf nodes
   1. Add edge from higher level to leaf node
3. Generate program according to the graph
   1. Based on the number of children each node has, select a function for it, and assign a data type for each node
   2. Run topological sort
   3. Assign variable name to each vertex in graph