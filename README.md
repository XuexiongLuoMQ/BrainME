# BrainEx
Brain Graph Explainer via Cross-Domain Meta-Learning for Brain Disorder Analysis

## Run
To perform our model, firstly pre-training:
```
python metapre.py --dataset =<dataset name>
```
`--dataset` is the name of the source dataset(such as PPMI)  
Then, test target task dataset:
```
python metapre.py --meta_test --dataset =<dataset name>
```
`--dataset` is the name of the task dataset(such as HIV) 


