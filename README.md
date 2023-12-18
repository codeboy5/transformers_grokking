### Pytorch (RE)-Implementation of Grokking Phenomenon

This is a pytorch re-implementation of [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).

I thought this would be a good paper to reproduce since this would allow me to code and train a GPT style model from scratch.

References used for the Code :-

1. [MinGPT by Karpathy](https://github.com/karpathy/minGPT)

#### Accuracy Loss Curves for Adam (with any weight decay)

<div style="display: flex; justify-content: space-between;">
  <img src="results/acc_40.0_adam.png" alt="Image 1" width="48%">
  <img src="results/loss_40.0_adam.png" alt="Image 2" width="48%">
</div>


#### Accuracy Loss Curves for AdamW ( &lambda; = 1 )

<div style="display: flex; justify-content: space-between;">
  <img src="results/acc_40.0_adamW_wdecay.png" alt="Image 1" width="48%">
  <img src="results/loss_40.0_adamW_wdecay.png" alt="Image 2" width="48%">
</div>
