# ProgrammableAging



## Reimplementation of Sir2-HAP Potential Landscape [in progress]


### Objectives

- Use  CNN to simulate stationary  Fokker-Planck Equation
- Speed up and exploring high dimension.


### Results and future work

The proposed method fails to simulate the FP stationary state.

- Issues
    - [ModelingToolkit.jl](https://mtk.sciml.ai/stable/) fails to apply chain rule for the partial differentiation. Therefore, I used SymPy.jl to [do the math](https://github.com/stevengogogo/ProgrammableAging/blob/c26c36bfa17dd34a35d8b674f8ca8af0a60953e4/src/sir2-hap-stationary.jl#L51) and figure out the FP equation.
    - [The script](https://github.com/stevengogogo/ProgrammableAging/blob/c26c36bfa17dd34a35d8b674f8ca8af0a60953e4/src/sir2-hap-stationary.jl) is compiled in single core - ~1 minute/step. 
        - Multiprocessing is not implemented yet.
        - [GPU process remains a issue](https://github.com/SciML/NeuralPDE.jl/issues/141) `2020/10/26`
    - The initial condition is not included in [the equation](https://github.com/stevengogogo/ProgrammableAging/blob/c26c36bfa17dd34a35d8b674f8ca8af0a60953e4/src/sir2-hap-stationary.jl#L65).


- Alternatives
    - Since the steady state can be easily derived from ODE, the SDE and potential landscape can be further derived from the minimum action as described in the article below:
        - > Tang, Ying, et al. ["Potential landscape of high dimensional nonlinear stochastic dynamics with large noise."](https://www.nature.com/articles/s41598-017-15889-2) Scientific reports 7.1 (2017): 1-11.

---

![]("https://imgur.com/abJpq67.gif")

**Video 1.** CNN estimated Potential landscape of Sir2-HAP model. (This is the preliminary version with compiling success but failed in using Multi-CPU and setting of initial values.) Produced by [sir2-hap-stationary.jl](https://github.com/stevengogogo/ProgrammableAging/blob/main/src/sir2-hap-stationary.jl).

---

### Reference
1. > Li, Yang, et al. ["A programmable fate decision landscape underlies single-cell aging in yeast."](https://science.sciencemag.org/content/369/6501/325) Science 369.6501 (2020): 325-329.
2. Repository: [ProgrammableAging](https://github.com/stevengogogo/ProgrammableAging)
3. [Slides](https://drive.google.com/file/d/1Fl3OGAg8WRCwlVqIRpAy9ERsougJxvxi/view?usp=sharing)
