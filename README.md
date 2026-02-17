# Conjugate and Neural Tangent Kernel Regression

This repository contains the implementations of the function and logistic regression tasks using Conjugate and Neural Tangent Kernels extracted from neural networks trained on the same tasks. For details, please refer to: 

**Qadeer, Saad, Andrew Engel, Amanda Howard, Adam Tsou, Max Vargas, Panos Stinis, and Tony Chiang. "Efficient kernel surrogates for neural network-based regression." arXiv preprint arXiv:2310.18612 (2023).**

+ The `NTKFuns.jl` and `PolyFuns.jl` files contain functions for the calculation of the NTK (described in the appendix/supplement), and polynomial-based operations respectively.
+ The `Fun_reg_ex.jl` script implements function regression for smooth functions, as described in Sections 4 and 6, and can be used to generate Figures 2 and 5.
+ The `Log_reg.ex.jl` script implements the logistic regression task, as described in Sections 5 and 6, and generates Figure 9.
