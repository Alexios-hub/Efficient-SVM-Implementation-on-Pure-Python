# Efficient-SVM-Implementation-on-Pure-Python
This is a python implementation of SVM based on SMO algorithm, and it is implemented according to the paper [Working Set Selection Using Second Order Informationfor Training Support Vector Machines](https://www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
# Installation

```
pip install -r requirements.txt
```

## Demo

See the demo.ipynb

## Optimization Problem

This algprithm is implemented to solve the optimization problem below:
$$
{\min_{w,b}\quad\frac{1}{2}|w|^{2}}\\\mathrm{s.t.}\quad y_{i}(w·x_{i}+b)-1\geqslant0,\quad i=1,2,\cdots,N
$$
Transform the problem to the dual form:
$$
\begin{aligned}
&\operatorname*{min}_{\alpha}\quad\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}(x_{i}\cdot x_{j})-\sum_{i=1}^{N}\alpha_{i} \\
&\mathrm{s.t.}\quad\sum_{i=1}^{N}\alpha_{i}y_{i}=0 \\
&\alpha_{i}\geqslant0,\quad i=1,2,\cdots,N
\end{aligned}
$$
