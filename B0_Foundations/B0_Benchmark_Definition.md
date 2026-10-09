# B0 — 1D Burgers PINN Benchmark Definition

## 1. Objective
This document freezes the mathematical problem, reference dataset, and evaluation metrics for the 1D viscous Burgers PINN study.
The objective is to develop a vanilla physics-informed neural network (PINN), validate its solution against a trusted reference dataset, and subsequently investigate how residual-point sampling affects solution accuracy.

## 2. Governing equation
We solve the one-dimensional viscous Burgers equation:

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu\frac{\partial^2u}{\partial x^2} = 0$$

where:
* $u(x,t)$ is the flow velocity.
* $x$ is the spatial coordinate.
* $t$ is time.
* $\nu$ is the viscosity/diffusion coefficient.

The benchmark uses:

$$\nu = \frac{0.01}{\pi}$$

The PDE residual used during PINN training is:

$$R_\theta(t,x) = \frac{\partial u_\theta}{\partial t} + u_\theta\frac{\partial u_\theta}{\partial x} - \nu\frac{\partial^2u_\theta}{\partial x^2}$$

The PDE loss penalizes this residual at the selected collocation points.

## 3. Computational domain
The benchmark domain is:

$$x \in [-1, 1], \qquad t \in [0, 1]$$

The neural network will take time and space as inputs, following the ordering used in the Raissi formulation:

$$(t,x) \longrightarrow u_\theta(t,x)$$

## 4. Initial condition
At $t=0$, the velocity profile is prescribed as:

$$u(x,0) = -\sin(\pi x)$$

This condition supplies known solution values at the initial time and is enforced through the initial-condition loss.

## 5. Boundary conditions
At both spatial boundaries:

$$u(-1,t) = 0, \qquad u(1,t) = 0$$

These conditions are enforced through the boundary-condition loss.

## 6. PINN training strategy
The vanilla PINN will be trained using three loss components:

$$L_{\mathrm{total}} = L_{\mathrm{PDE}} + L_{\mathrm{IC}} + L_{\mathrm{BC}}$$

* **PDE loss:** penalizes violations of the governing equation at collocation points.
* **Initial-condition loss:** penalizes errors in the prescribed initial velocity profile.
* **Boundary-condition loss:** penalizes errors at the two spatial boundaries.

The reference solution will not be used to train the vanilla PINN. It will be reserved for evaluating the trained solution.

## 7. Reference dataset
The selected reference dataset is Burgers.npz, obtained from the repository accompanying Wu et al. (2023):
https://github.com/lu-group/pinn-sampling

The arrays in the supplied dataset are:

| Array | Shape | Meaning |
| :--- | :--- | :--- |
| `x` | (256, 1) | Spatial coordinates |
| `t` | (100, 1) | Reference time coordinates |
| `usol` | (256, 100) | Reference velocity field |

The reference time coordinates run from $t=0$ to $t=0.99$ in increments of $0.01$. Therefore, evaluation against this dataset will cover its available time grid, while the PINN's prescribed domain remains $t \in [0, 1]$.

The array orientation will be checked explicitly in the implementation before calculating errors.

## 8. Evaluation metrics
Primary metric: relative L2 error

$$E_{L_2} = \frac{\Vert{}u_{\mathrm{PINN}} - u_{\mathrm{ref}}\Vert{}_2}{\Vert{}u_{\mathrm{ref}}\Vert{}_2}$$

The norm is evaluated over the reference evaluation grid.

Supplementary metric: relative maximum error

$$E_{L_\infty} = \frac{\max \vert{}u_{\mathrm{PINN}} - u_{\mathrm{ref}}\vert{}}{\max \vert{}u_{\mathrm{ref}}\vert{}}$$

This supplementary metric helps identify large local errors that might not be obvious from the overall relative L2 error.

The PDE, initial-condition, and boundary-condition losses will also be reported separately.

## 9. Mathematical reference and provenance
The mathematical formulation follows the Burgers benchmark described by Raissi et al. (2019) and Wu et al. (2023).

The Hopf–Cole transformation provides the mathematical basis for relating Burgers' equation to a linear heat equation. The worked example in Kadalbajoo and Awasthi (2006) uses a different spatial domain and initial condition, so its specific example formula will not be assumed to be the reference solution for this benchmark.

The numerical reference used in this project is the dataset distributed with the Wu et al. repository. No claim is made here that Burgers.npz is a byte-for-byte conversion of Raissi's burgers_shock.mat.

## 10. Planned progression
1. B0 — Understand the formulation and freeze the benchmark.
2. B1 — Implement a vanilla PyTorch Burgers PINN.
3. B2 — Evaluate the learned solution across the space-time domain.
4. B3 — Validate against the reference dataset.
5. B4 — Compare selected uniform and residual-based adaptive sampling methods.

**Status:** Benchmark formulation defined; implementation and numerical validation are pending.

## References
* Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
* Wu, C., Zhu, M., Tan, Q., Kartha, Y., and Lu, L. (2023). A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 403, 115671. https://doi.org/10.1016/j.cma.2022.115671
* Kadalbajoo, M. K., and Awasthi, A. (2006). A numerical method based on Crank–Nicolson scheme for Burgers' equation. Applied Mathematics and Computation, 182, 1430–1442. https://doi.org/10.1016/j.amc.2006.05.030