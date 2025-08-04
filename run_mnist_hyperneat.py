import sys
import os

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT

import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.common import ACT, AGG
from problems.mnist_problem import MNISTClassificationProblem

problem = MNISTClassificationProblem()

import jax.nn as jnn

# # Hidden coords normalized over [-1, 1], same scale as input
# hidden_coors = [(x / 9 * 2 - 1, y / 9 * 2 - 1) for y in range(10) for x in range(10)]
# # hidden_coors.append((0.0, -1.2))  # Add bias for hidden layer if desired

# # Output coords normalized over [-1, 1]
# output_coors = [(x / 9 * 2 - 1, 0.0) for x in range(10)]
# # Optionally add bias to output layer if needed

# input_coors = [
#     (x / 4.0 - 1.0, y / 4.0 - 1.0)
#     for y in range(8)
#     for x in range(8)
# ]
# input_coors.append((0.0, -1.2))  # Bias for input layer

# substrate = FullSubstrate(
#     input_coors=input_coors,
#     hidden_coors=hidden_coors,
#     output_coors=output_coors,
# )

# Input coords for 28x28 input layer, normalized to [-1, 1]
input_coors = [
    (x / 27 * 2 - 1, y / 27 * 2 - 1)  # divide by 27 since indices run 0..27
    for y in range(28)
    for x in range(28)
]
input_coors.append((0.0, -1.2))  # Bias for input layer (optional)

# Hidden coords (for example 14x14 grid) normalized over [-1, 1]
hidden_size = 14
hidden_coors = [
    (x / (hidden_size - 1) * 2 - 1, y / (hidden_size - 1) * 2 - 1)
    for y in range(hidden_size)
    for x in range(hidden_size)
]

# Output coords normalized over [-1, 1], 10 outputs aligned on x axis at y=0
output_size = 10
output_coors = [
    (x / (output_size - 1) * 2 - 1, 0.0)
    for x in range(output_size)
]

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors,
)


pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        weight_threshold=.1,
        neat=NEAT(
            pop_size=500,
            species_size=30,
            survival_threshold=0.2,
            compatibility_threshold=5,
            genome=DefaultGenome(
                num_inputs=4,
                num_outputs=1,
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.sin, ACT.sigmoid, ACT.gauss, ACT.identity],
                    aggregation_options=[AGG.sum]
                ),
                max_nodes=100,
                max_conns=500,
                init_hidden_layers=(5,),
                output_transform=ACT.tanh,
            ),
        ),
        activation=ACT.tanh,
        activate_time=2,
        output_transform=jnn.softmax,
    ),
    problem=problem,
    generation_limit=2000,
    fitness_target=0.8,
    seed=42,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)
