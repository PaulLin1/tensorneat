import sys
import os
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG
from problems.mnist_problem import MNISTClassificationProblem  # ← your custom problem file
import jax.numpy as jnp
# Use MNIST instead of Digits
problem = MNISTClassificationProblem()
algorithm=NEAT(
    pop_size=50,
    species_size=30,
    survival_threshold=0.2,
    genome=DefaultGenome(
        num_inputs=784,  # ← MNIST input (28x28)
        num_outputs=10,
        # init_hidden_layers=(),
        # node_gene = BiasNode(
        #     activation_options=[ACT.relu],  # only identity activation
        #     aggregation_options=[AGG.sum],       # only sum aggregation
        # ),
        max_nodes=2000,
        max_conns=10000,
        output_transform=ACT.relu,
    ),
)

pipeline = Pipeline(
    algorithm=algorithm,
    problem=problem,
    generation_limit=100,
    fitness_target=.6,
    seed=42,
    is_save=True, save_dir='models/'
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)

