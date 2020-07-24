# Max-Min Margin Markov Networks (M<sup>4</sup>Ns)
Code accompanying [Consistent Structured Prediction with Max-Min Margin Markov Networks](https://arxiv.org/pdf/2007.01012.pdf) published at ICML 2020

## Structured Prediction

## Difference between M<sup>3</sup>Ns ([Taskar et al., 2004](https://papers.nips.cc/paper/2397-max-margin-markov-networks.pdf)) or Structural SVM ([Tsochantaridis et al., 2005](http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)) and M<sup>4</sup>Ns

## Reproduce Experiments

### Compile project
Run the Makefile:
```
make all
```

### Datasets for Multi-class Classification, Ordinal Regression, Sequence Prediction and Ranking
Get the datasets folder from the following link [link to drive](a). Copy folder inside ```struntho/datasets/```


### Run experiments
Run: 
```
python main.py \
   --task [options: multiclass/ordinal/sequence/ranking]  # choose structured prediction task
   --model [options: m3n/crf/m4n]  # choose model to run
   --dataset [satimage (example from multilcass)]  # set an available dataset for the chosen task 
   --add_bias  # add bias to the linear model
   --cython  # use the compiled version for fast inference
   --reg [float]  # regularization parameter lambda
   --check_dual_every [int]  # frequency of computing the dual gap
   --iter_oracle [int]  # iterations of saddle-point mirror-prox to compute the oracle
   --kernel  # not applicable to sequence prediction
   --epochs [int]  # number of passes over the training dataset
```
As an example:
```
python main.py --task multiclass --model m4n --dataset segment --add_bias --cython --reg 0.001 --check_dual_every 10 --iter_oracle 20 --kernel --epochs 101
```
>>>>>>> fe3aef032eca408eafc16470a170e34dd6c4e3e0
