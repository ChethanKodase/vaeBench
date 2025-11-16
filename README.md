# vaeBench
This project involves the analysis of manifolds learned by autoencoders.


### Training Beta-VAEs for different values of beta:


`export CUDA_VISIBLE_DEVICES=0`

`conda activate vaeBench`

`python trainBetaVAE.py --latentDimesion 2 --learningRate 1e-4 --numEpochs 100 --dataset MNIST --betaValue 0.0 --seedVal 0`

