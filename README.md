# GM2020

This repo contains some deep learning experiments with chess principles as targets for learning.

chess.ipynb is where the deep learning stuff is, and positions.cc is code to generate training data in the form of chess positions as input and various concepts associated with them as output.

In naivePositions.cc, which I am currently using to generate my training data, the output is a matrix representing a simple concept - the squares on the diagonals that any white bishops rest on. For example: 

```
RNBQKBNR
PPPPPPPP
........
........
........
........
pppppppp
rnbqkbnr
```
If this is starting position, with capital letters representing white's pieces, then:
```
........
.X.XX.X.
X..XX..X
..X..X..
.X....X.
X......X
........
........
```
This is a representation of the expected output, marking the white bishops' diagonals.

The goal of the project is to find algorithms that build complex concepts as a function of simpler concepts, and that can iteratively continue to build complexity. Using simpler tasks to start with has allowed me to explore a domain where I can clearly follow the path the data takes through the network, and the goal is to maintain that transparency as the network scales.
