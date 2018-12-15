# Visualizing Quantum Dynamics

OpenGL visualization of animated quantum dynamics (QD) simulation of a 2-dimensional (2D) electronic wave function in real time.

## File description

* qd.c - Single thread quantum dynamics simulations
* atomv.c - Visualizing the molecular dynamic process with OpenGL
* animqd.cu - Visualizing the quantum dynamic process with OpenGL

## Running the tests

```
gcc atomv.c -o atomv -lm -lGL -lGLU -lglut
gcc animqd.c -o animqd -lm -lGL -lGLU -lglut
```
