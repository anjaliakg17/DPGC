# Releasing Graph Neural Networks with Differential Privacy Guarantees
<!--### by Iyiola E. Olatunji, Thorben Funke, Megha Khosla-->
This repository contains additional details and reproducibility of the stated results. 

## Motivation: 
Real-world graphs, such often are associated with sensitive information about individuals and their activities. Hence, they cannot always be made public. Moreover, graph neural networks (GNNs) models trained on such sensitive data can leak significant amount of information e.g via membership inference attacks.

## Goal of the Paper: 
Release a GNN model that is trained on sensitive data yet robust to attacks. Importantly, it should have differential privacy (DP) guarantees.


## Reproducing Results
PrivGNN can be executed via 
```
python3 knn_graph.py
```

### Changing Parameters & Setting
Please adjust the `all_config.py` file to change parameters, such as the datasets, K, lambda, attack, baselines, or gamma.
