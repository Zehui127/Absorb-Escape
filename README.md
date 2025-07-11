# Absorb-Escape

Absorb \& Escape: Overcoming Single Model Limitations in Generating Genomic Sequences (Neural Information Processing Systems, 2024) [Paper Link](https://arxiv.org/abs/2410.21345)

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
& \quad\quad \text{/* Absorb Step */} \\
& \quad\quad j = i + 1 \\
& \quad\quad \tilde{x}'_j \sim p^{AR}(x_j | x_{0:i}) \\
\\
& \quad\quad \text{while } p^{AR}(\tilde{x}'_j) > p^{DM}(\tilde{x}_j): \\
& \quad\quad\quad j = j + 1 \\
& \quad\quad\quad \text{/* Refine inaccurate region token by token */} \\
& \quad\quad\quad \tilde{x}'_j \sim p^{AR}(x_j | x_{0:i}, x_{i:j-1}) \\
\\
& \quad\quad \text{/* Escape Step */} \\
& \quad\quad \tilde{x}_{i:j} = \tilde{x}'_{i:j} \\
& \quad\quad i = i + j \\
\\
\textbf{Output:} \\
& \bullet \text{ Refined sequence } \tilde{x} \text{ with improved quality}
\end{align*}
```

## Reproduction of Experiment
### Transcription Profile conditioned Promoter Design

We have implemented A\&E with the codebase of [Dirichlet Flow Matching with Applications to DNA Sequence Design](https://github.com/HannesStark/dirichlet-flow-matching/tree/main). The installation of depdenencies and evaluation data are the same as the original repository. The only difference is that an additional checkpoint file needs to be downloaded. For running the experiment, please follow the instructions below to reproduce the experiment:

Step1. Installing the required package
```yaml
### Conda environment
conda create -c conda-forge -n seq python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric jupyterlab gpustat pyyaml wandb biopython spyrmsd einops biopandas plotly seaborn prody tqdm lightning imageio tmtools "fair-esm[esmfold]" e3nn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu113.htm

# The libraries below are required for the promoter design experiments
git clone https://github.com/kathyxchen/selene.git
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install
pip install pyBigWig pytabix cooler pyranges biopython cooltools
```
Step2. Download the reference genome and the evaluation data from https://zenodo.org/records/7943307. Place it in `DFM-with-Absorb-Escape/data`. Or you need to edit the path config in `DFM-with-Absorb-Escape/utils/promoter_dataset.py` and line 29 of `DFM-with-Absorb-Escape/lightning_modules/promoter_module_refine.py`to point to the correct path

Step3. Download the checkpoint of the pretrained model `ae.ckpt` from [here](https://huggingface.co/Zehui127127/Absorb-Escape/tree/main), this checkpoint is simply the combination of pretrained **DFM distilled** and **AR model** by the original DFM paper.

Step4. Run inference on the test set, where the --ckpt shoud be the path of downloaded checkpoint file `ae.ckpt`. This will run the Absorb-Escape algorithm on the test set and generate the refined sequences.

```bash
cd DFM-with-Absorb-Escape
python -m train_promo --run_name dirichlet_flow_matching_distilled --batch_size 128 --wandb --num_workers 4 --num_integration_steps 100 --ckpt workdir/ae.ckpt --validate --validate_on_test --mode distill
```



### Multi-species DNA Sequences Generation
Please See https://github.com/Zehui127/Latent-DNA-Diffusion and https://github.com/Genentech/regLM, where we implement the A\&E algrithm based on these two algorithms. 
