# For following baselines
[VLN:'variation_neighborhoods',
LVE: 'variation_edges',
LVC: 'variation_cliques',
HEM: 'heavy_edge',
Alg. Distance: 'algebraic_JC',
Affinity:  'affinity_GS',
Korn: 'kron']

# Change the method "\SCAL\graph_coarsening\coarsening_utils.py" e.g,

method="variation_neighborhood",

[VLN:'variation_neighborhoods',
LVE: 'variation_edges',
LVC: 'variation_cliques',
HEM: 'heavy_edge',
Alg. Distance: 'algebraic_JC',
Affinity:  'affinity_GS',
Korn: 'kron']

#example, the coarsening ratio is 0.5

python train.py --dataset [] --experiment random --epoch 50 --coarsening_ratio 0.5
