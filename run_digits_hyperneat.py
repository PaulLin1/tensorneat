import sys
import os

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem
from tensorneat.genome import DefaultGenome, BiasNode
import jax.nn as jnn
problem = DigitsClassificationProblem()

# Normalize input coordinates consistently over [-1,1]
input_coors = [(x / 7 * 2 - 1, y / 7 * 2 - 1) for y in range(8) for x in range(8)]
input_coors.append((0.0, -1.2))  # Bias for input layer

# Hidden coords normalized over [-1, 1], same scale as input
hidden_coors = [
    (x / 20 * 2 - 1, -0.4)
    for x in range(20)
]
# hidden_coors.append((0.0, -1.2))  # Add bias for hidden layer if desired

# Output coords normalized over [-1, 1]
output_coors = [(x / 9 * 2 - 1, 0.0) for x in range(10)]
# Optionally add bias to output layer if needed

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors,
)

pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        weight_threshold=.2,  # increase to prune weak links
        neat=NEAT(
            pop_size=500,          # smaller but still decent population
            species_size=30,       # more balanced speciation
            survival_threshold=0.01,  # keep top 20% survive to maintain diversity
            genome=DefaultGenome(
                num_inputs=4,      # CPPN inputs: (x1, y1, x2, y2)
                num_outputs=3,
                max_nodes=100,
                max_conns=300,
                init_hidden_layers=(5,),  # start simple, add layers via mutation
                output_transform=ACT.tanh,
            ),
        ),
        activation=ACT.tanh,
        activate_time=15,
        output_transform=jnn.softmax,
    ),
    problem=problem,
    generation_limit=2000,
    fitness_target=0.9,   # set reasonable target to encourage progress
    seed=42,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)
