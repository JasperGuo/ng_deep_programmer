## Random Program Generator

#### Basic Idea:

Randomly build a data flow graph, and then assign input nodes data types, and based on the data types and the number of input edges select functions.

#### Node Types

| Node Type     | # input edges | # output edges |
| ------------- | ------------- | -------------- |
| Input Node    | 0             | NOT LIMIT      |
| Output Node   | {1, 2, 3}     | 0              |
| Internal Node | {1, 2, 3}     | 1              |

#### Constrain

1. **DAG**
2. No edges between input nodes
3. Edge Must from lower level to higher level