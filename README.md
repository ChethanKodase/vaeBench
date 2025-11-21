# vaeBench
This project involves the analysis of manifolds learned by autoencoders.

In these experiments we use `pythae` library to get all the benchmark variational autoenocders. Install `pythae` library from https://pypi.org/project/pythae/ using the below commands

`git clone https://github.com/clementchadebec/benchmark_VAE.git`

`cd benchmark_VAE`

`pip install -e .`


### Training Beta-VAEs for different values of beta:


`export CUDA_VISIBLE_DEVICES=0`

`conda activate vaeBench`

`python trainBetaVAE.py --latentDimesion 2 --learningRate 1e-4 --numEpochs 100 --dataset MNIST --betaValue 0.0 --seedVal 0`


### To get cluster purity plots for trained Beta- VAEs
 
```
export CUDA_VISIBLE_DEVICES=0
cd vaeBench/betaVAE
conda activate vaeBench
python clusterPurityBetaVAE.py --learningRate 1e-4 --numEpochs 100 --dataset MNIST --seedVal 0 --whichBetas big
python clusterPurityBetaVAE.py --learningRate 1e-4 --numEpochs 100 --dataset MNIST --seedVal 0 --whichBetas small
python clusterPurityBetaVAE.py --learningRate 1e-4 --numEpochs 100 --dataset MNIST --seedVal 0 --whichBetas all

```