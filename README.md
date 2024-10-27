# Absorb-Escape
Absorb&amp; Escape: A generalizable sampling algorithm for DNA generation combining diffusion model and auto-regressive model

[Paper Link](https://neurips.cc/virtual/2024/poster/94782) (Neural Information Processing Systems, 2024)

## Standalone Implementation of Fast Absorb-Escape

A standalone version of the Fast Absorb-Escape algorithm can be found in `AE_sampler.py`. Below is the pseudocode to help implement the Fast Absorb-Escape algorithm:

```math
\begin{align*}
\textbf{Input:} \\
& \bullet \text{ Absorb Threshold } (T_{Absorb}) \\
& \bullet \text{ Pretrained AutoRegressive model } (p^{AR}) \\
& \bullet \text{ Pretrained Diffusion Model } (p^{DM}) \\
\\
\textbf{Algorithm:} \\
& 1. \text{ Initialize } \tilde{x}^0 \sim p^{DM}(x) \\
\\
& 2. \text{ for } i = 0 \text{ to } len(\tilde{x}): \\
& \quad \text{if } p^{DM} < T_{Absorb}: \\
& \quad\quad \text{# Absorb Step} \\
& \quad\quad j = i + 1 \\
& \quad\quad \tilde{x}'_j \sim p^{AR}(x_j | x_{0:i}) \\
\\
& \quad\quad \text{while } p^{AR}(\tilde{x}'_j) > p^{DM}(\tilde{x}_j): \\
& \quad\quad\quad j = j + 1 \\
& \quad\quad\quad \text{# Refine inaccurate region token by token} \\
& \quad\quad\quad \tilde{x}'_j \sim p^{AR}(x_j | x_{0:i}, x_{i:j-1}) \\
\\
& \quad\quad \text{# Escape Step} \\
& \quad\quad \tilde{x}_{i:j} = \tilde{x}'_{i:j} \\
& \quad\quad i = i + j \\
\\
\textbf{Output:} \\
& \bullet \text{ Refined sequence } \tilde{x} \text{ with improved quality}
\end{align*}
```

## Reproduction of Experiment
### Transcription Profile conditioned Promoter Design

We have implemented A\&E with the codebase of [Dirichlet Flow Matching with Applications to DNA Sequence Design](https://github.com/HannesStark/dirichlet-flow-matching/tree/main). The installation of depdenencies and evaluation data are the same as the original repository. The only difference is that checkpoint an additional checkpoint file needs to be downloaded. For running the experiment, please follow the instructions below to reproduce the experiment:

### Toy Example (on Synthetic Data generated from Hidden Markov Model)


### Multi-species DNA Sequences Generation
