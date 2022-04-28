# Image Patching

## Brief
The script gives a draft solution to the Image Patching problem

## Problem Formulation
The source data of the algorithm are three images
* first source `u1`
* second source `u2`
* binary mask `m`

The problem is to find a new image `v` which looks like `u1` at regions where the mask `m` is zero, and looks like `u2` at the rest regions. 

## Solution
The main idea of the algorithm is to reformulate the original problem as an optimization problem

<img src="https://latex.codecogs.com/svg.image?v=\arg\min_{v}\iint\left(v\left(x,y\right)-u_{1}\left(x,y\right)\right)^{2}&plus;\lambda\left\Vert&space;\nabla&space;u-g\left(x,y\right)\right\Vert&space;^{2}dxdy" title="https://latex.codecogs.com/svg.image?v=\arg\min_{v}\iint\left(v\left(x,y\right)-u_{1}\left(x,y\right)\right)^{2}+\lambda\left\Vert \nabla u-g\left(x,y\right)\right\Vert ^{2}dxdy" />

where `λ` is a parameters of the algorithm, and the vector field `g`

<img src="https://latex.codecogs.com/svg.image?g\left(x,y\right)=\begin{cases}\nabla&space;u_{1}\left(x,y\right)&space;&&space;\text{if&space;}m\left(x,y\right)=0\\\nabla&space;u_{2}\left(x,y\right)&space;&&space;\text{otherwise}\end{cases}" title="https://latex.codecogs.com/svg.image?g\left(x,y\right)=\begin{cases}\nabla u_{1}\left(x,y\right) & \text{if }m\left(x,y\right)=0\\\nabla u_{2}\left(x,y\right) & \text{otherwise}\end{cases}"/>

is constructed from the gradients of the source images `∇u1` and `∇u2`.

The solution to the optimization problem is a solution of the Euler-Lagrange PDE

<img src="https://latex.codecogs.com/svg.image?\lambda\triangle&space;v-v&space;=&space;\lambda\nabla\cdot&space;g&space;&plus;&space;u_1" title="https://latex.codecogs.com/svg.image?\lambda\triangle v-v = \lambda\nabla\cdot g + u_1"/>

The last one is solved with the help of Fourier transform.

# Example
There are three inputs

| Source Image | Patch Image | Mask |
|:---:|:---:|:---:|
| [<img src="./data/source.png" width="250"/>](./data/source.png) | [<img src="./data/patch.png" width="250"/>](./data/patch.png) | [<img src="./data/mask.png" width="250"/>](./data/mask.png) |

and the result of the algorithms campated to the usual copy by the mask

| Result of optimization | Simple copy | 
|:---:|:---:|
| [<img src="./data/out-merge.png" width="250"/>](./data/out-merge.png) | [<img src="./data/out-copy.png" width="250"/>](./data/out-copy.png)
