# Absorb-Escape
Absorb&amp; Escape: A generalizable sampling algorithm for DNA generation combining diffusion model and auto-regressive model

[Paper Link](https://neurips.cc/virtual/2024/poster/94782) (Neural Information Processing Systems, 2024)

## Standalone Implementation of Fast Absorb-Escape

## Fast Absorb-Escape Algorithm

A standalone version of the Fast Absorb-Escape algorithm can be found in `AE_sampler.py`. Below is the pseudocode to help implement the Fast Absorb-Escape algorithm:

```pseudo
Input:
- Absorb Threshold (T_Absorb)
- Pretrained AutoRegressive model (p_AR)
- Pretrained Diffusion Model (p_DM)

Algorithm:
1. Initialize \tilde{x}^0 \sim p_{DM}(x)

2. for i = 0 to len(\tilde{x}):
       if p_{DM} < T_{Absorb}:
           # Absorb Step
           j = i + 1
           \tilde{x}'_j \sim p_{AR}(x_j \mid x_{0:i})

           while p_{AR}(\tilde{x}'_j) > p_{DM}(\tilde{x}_j):
               j = j + 1
               # Refine inaccurate region token by token
               \tilde{x}'_j \sim p_{AR}(x_j \mid x_{0:i}, x_{i:j-1})

           # Escape Step
           \tilde{x}_{i:j} = \tilde{x}'_{i:j}
           i = i + j

Output:
- Refined sequence \tilde{x} with improved quality
```

## Reproduction of Experiment
### Transcription Profile conditioned Promoter Design

We have implemented A\&E with the codebase of [Dirichlet Flow Matching with Applications to DNA Sequence Design](https://github.com/HannesStark/dirichlet-flow-matching/tree/main). The installation of depdenencies and evaluation data are the same as the original repository. The only difference is that checkpoint an additional checkpoint file needs to be downloaded. For running the experiment, please follow the instructions below to reproduce the experiment:

### Toy Example (on Synthetic Data generated from Hidden Markov Model)


### Multi-species DNA Sequences Generation
