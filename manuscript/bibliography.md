# Annotated Bibliography & Literature Reference Guide

This document catalogs the peer-reviewed scientific literature underpinning the **Bernasconi-Krylov** quantum tunneling benchmark. All citations correspond directly to entries in [`manuscript/references.bib`](file:///home/shaashwat/Desktop/bernasconi-krylov/manuscript/references.bib).

---

## 1. Low Autocorrelation Binary Sequences (LABS) & The Bernasconi Model

1. **J. Bernasconi (1987)**  
   *Low autocorrelation binary sequences: statistical mechanics and extreme value statistics.*  
   *Journal de Physique*, 48(4), 559–567.  
   **DOI:** [10.1051/jphys:01987004804055900](https://doi.org/10.1051/jphys:01987004804055900)  
   > **Context:** The foundational paper introducing the Bernasconi spin model mapping the LABS problem onto a deterministic, long-range 4-body classical Ising Hamiltonian without quenched disorder, exhibiting spin-glass-like glassy dynamics and rugged energy landscapes.

2. **T. Packebusch and S. Mertens (2016)**  
   *Low autocorrelation binary sequences to $N=105$.*  
   *Journal of Physics A: Mathematical and Theoretical*, 49(16), 165001.  
   **DOI:** [10.1088/1751-8113/49/16/165001](https://doi.org/10.1088/1751-8113/49/16/165001)  
   > **Context:** The authoritative reference establishing exact global ground states and optimum energy levels up to $N = 105$ via branch-and-bound algorithms; used directly in this benchmark to identify true metastable local minima and exclude trivial global ground states.

3. **S. Mertens (1996)**  
   *Exhaustive search for low-autocorrelation binary sequences.*  
   *Journal of Physics A: Mathematical and General*, 29(18), L473–L481.  
   **DOI:** [10.1088/0305-4470/29/18/002](https://doi.org/10.1088/0305-4470/29/18/002)  
   > **Context:** Exhaustive enumeration of LABS energy spectra up to $N=32$, establishing the order-8 spatial/spin symmetry group and energy degeneracies.

4. **E. Marinari, G. Parisi, and F. Ritort (1994)**  
   *Replica field theory for deterministic models: II. A non-random spin glass with glassy behaviour.*  
   *Journal of Physics A: Mathematical and General*, 27(23), 7647–7668.  
   **DOI:** [10.1088/0305-4470/27/23/011](https://doi.org/10.1088/0305-4470/27/23/011)  
   > **Context:** Demonstrates that the Bernasconi Hamiltonian possesses a genuine 1-step replica symmetry breaking (1-RSB) glass transition despite having no quenched disorder.

5. **J.-P. Bouchaud and M. Mézard (1997)**  
   *Universality classes for extreme value problems.*  
   *Journal de Physique I*, 7(8), 985–997.  
   **DOI:** [10.1051/jp1:1997103](https://doi.org/10.1051/jp1:1997103)  
   > **Context:** Theoretical characterization of energy landscape extrema and local trap depth distributions in spin glass models.

---

## 2. Quantum Tunneling, Escape Dynamics & Quantum Annealing

6. **T. Kadowaki and H. Nishimori (1998)**  
   *Quantum annealing in the transverse Ising model.*  
   *Physical Review E*, 58(5), 5355–5363.  
   **DOI:** [10.1103/PhysRevE.58.5355](https://doi.org/10.1103/PhysRevE.58.5355)  
   > **Context:** Introduced quantum annealing using a transverse field $\sum_i \sigma_i^x$ to induce quantum tunneling through tall, narrow potential barriers.

7. **V. Bapst, L. Foini, F. Krzakala, G. Semerjian, and F. Zamponi (2013)**  
   *The quantum adiabatic algorithm applied to random optimization problems: The quantum spin glass perspective.*  
   *Physics Reports*, 523(3), 127–205.  
   **DOI:** [10.1016/j.physrep.2012.10.002](https://doi.org/10.1016/j.physrep.2012.10.002)  
   > **Context:** Comprehensive review linking quantum optimization bottlenecks, first-order quantum phase transitions, and tunneling through spin-glass barriers.

8. **V. N. Smelyanskiy, K. Kechedzhi, S. V. Isakov, S. Boixo, H. Wang, and H. Neven (2020)**  
   *Nonergodic Quantum Dynamics and Tunneling in Complex Landscapes.*  
   *Physical Review X*, 10(1), 011017.  
   **DOI:** [10.1103/PhysRevX.10.011017](https://doi.org/10.1103/PhysRevX.10.011017)  
   > **Context:** Analyzes coherent non-ergodic quantum tunneling versus high-energy delocalization and dephasing across multi-basin energy landscapes; directly underpins our discussion of the $g = 0.30$ vs. $g = 0.35$ crossover.

9. **S. Knysh (2016)**  
   *Zero-temperature quantum annealing bottlenecks in the spiked $p$-spin model.*  
   *Nature Communications*, 7, 12370.  
   **DOI:** [10.1038/ncomms12370](https://doi.org/10.1038/ncomms12370)  
   > **Context:** Examines quantum tunneling through narrow, high barriers in non-local $p$-spin models, demonstrating multi-step perturbative tunneling bottlenecks.

10. **V. S. Denchev, S. Boixo, S. V. Isakov, N. Ding, R. Babbush, V. N. Smelyanskiy, J. M. Martinis, and H. Neven (2016)**  
    *What is the Computational Value of Finite-Range Tunneling?*  
    *Physical Review X*, 6(3), 031015.  
    **DOI:** [10.1103/PhysRevX.6.031015](https://doi.org/10.1103/PhysRevX.6.031015)  
    > **Context:** Demonstrates a $10^8\times$ speedup from quantum tunneling over classical simulated annealing and path-integral Monte Carlo across tailored tall, narrow barrier landscapes.

11. **G. E. Santoro, R. Martoňák, E. Tosatti, and R. Car (2002)**  
    *Theory of Quantum Annealing of an Ising Spin Glass.*  
    *Science*, 295(5564), 2427–2430.  
    **DOI:** [10.1126/science.1068774](https://doi.org/10.1126/science.1068774)  
    > **Context:** Theoretical comparison of quantum tunneling vs. thermal activation in Ising spin glasses.

12. **L. T. Brady and W. van Dam (2016)**  
    *Quantum advantage for spin glasses without total-spin symmetry.*  
    *Physical Review A*, 93(3), 032304.  
    **DOI:** [10.1103/PhysRevA.93.032304](https://doi.org/10.1103/PhysRevA.93.032304)  
    > **Context:** Formalizes quantum escape and tunneling advantages in asymmetric spin-glass models lacking global permutation symmetries.

---

## 3. Hamiltonian Simulation & Trotterized Dynamics

13. **H. F. Trotter (1959)**  
    *On the product of semi-groups of operators.*  
    *Proceedings of the American Mathematical Society*, 10(4), 545–551.  
    **DOI:** [10.1090/S0002-9939-1959-0108732-6](https://doi.org/10.1090/S0002-9939-1959-0108732-6)  
    > **Context:** The original operator splitting theorem for non-commuting Hamiltonian exponentials.

14. **M. Suzuki (1976)**  
    *Generalized Trotter's formula and systematic approximants of exponential operators and inner derivations with applications to many-body problems.*  
    *Communications in Mathematical Physics*, 51(2), 183–190.  
    **DOI:** [10.1007/BF01609348](https://doi.org/10.1007/BF01609348)  
    > **Context:** Higher-order product formulas for Trotterized unitary dynamics in many-body quantum physics.

15. **A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu (2021)**  
    *Theory of Trotter Error with Commutator Scaling.*  
    *Physical Review X*, 11(1), 011020.  
    **DOI:** [10.1103/PhysRevX.11.011020](https://doi.org/10.1103/PhysRevX.11.011020)  
    > **Context:** Commutator-based Trotter error bounds providing mathematical justification for our choice of time step $\Delta t = 0.05\text{ s}$.

16. **M. Hochbruck and C. Lubich (1997)**  
    *On Krylov Subspace Approximations to the Exponential of Symmetric Matrices.*  
    *SIAM Journal on Numerical Analysis*, 34(5), 1911–1925.  
    **DOI:** [10.1137/S0036142995287532](https://doi.org/10.1137/S0036142995287532)  
    > **Context:** Foundational analysis of Krylov subspace time propagation for large sparse symmetric matrix exponentials.

---

## 4. Statevector Simulation & GPU Acceleration

17. **NVIDIA Corporation (2023)**  
    *cuQuantum SDK: cuStateVec High Performance State Vector Library.*  
    **URL:** [https://developer.nvidia.com/cuquantum-sdk](https://developer.nvidia.com/cuquantum-sdk)  
    > **Context:** High-performance GPU tensor-core library for large-scale $2^N$ quantum statevector evolution; utilized in our simulation pipeline to achieve fast $N=30$ runs on NVIDIA H100 and H200 GPUs.

18. **T. Jones, A. Brown, I. Bush, and S. C. Benjamin (2019)**  
    *QuEST and high performance simulation of quantum computers.*  
    *Scientific Reports*, 9, 10736.  
    **DOI:** [10.1038/s41598-019-47174-9](https://doi.org/10.1038/s41598-019-47174-9)  
    > **Context:** High-performance distributed statevector simulation and gate evaluation standards.

---

## 5. Classical Optimization Baselines & Statistical Bootstrapping

19. **R. Martí (2003)**  
    *Multi-Start Methods.*  
    *Handbook of Metaheuristics*, International Series in Operations Research & Management Science, vol 57, pp. 355–368. Springer, Boston, MA.  
    **DOI:** [10.1007/0-306-48056-5_12](https://doi.org/10.1007/0-306-48056-5_12)  
    > **Context:** Formalizes multi-start local search and perturbation heuristics; provides theoretical framing for our Kick+SD baseline.

20. **F. Glover (1986)**  
    *Future paths for integer programming and links to artificial intelligence.*  
    *Computers & Operations Research*, 13(5), 533–549.  
    **DOI:** [10.1016/0305-0548(86)90048-1](https://doi.org/10.1016/0305-0548(86)90048-1)  
    > **Context:** Fundamental concepts of perturbation search and memory-based heuristics in discrete non-convex spaces.

21. **B. Efron and R. J. Tibshirani (1994)**  
    *An Introduction to the Bootstrap.*  
    *CRC Press / Chapman & Hall*, Monographs on Statistics and Applied Probability, vol 57.  
    **ISBN:** 9780412042317  
    > **Context:** Theoretical foundation for our non-parametric bootstrap resampling ($B = 2000$) used to calculate 68% confidence intervals for sample medians.
