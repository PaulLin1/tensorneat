import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT

import jax.numpy as jnp
import jax.nn as jnn

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem
problem = DigitsClassificationProblem()


from tensorneat.algorithm.hyperneat import FullSubstrate

input_coors = [(x / 3.5 - 1.0, y / 3.5 - 1.0) for y in range(8) for x in range(8)]  # 64
input_coors.append((0.0, -1.2))  # Bias


hidden_coors = [
    ((x / 5.0 - 1.0), -0.4) for x in range(10)
] + [
    ((x / 5.0 - 1.0), 0.0) for x in range(10)
]
output_coors = [(x / 5.0 - 0.9, 1.0) for x in range(10)]  # 10 output classes



substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors,
)

pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        weight_threshold=0.005,
        neat=NEAT(
            pop_size=500,
            species_size=20,
            survival_threshold=0.1,
            genome=DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                # init_hidden_layers=(),
                init_hidden_layers=(5,),  # add hidden layer
                # activation_set=['sigmoid', 'tanh', 'gauss', 'sin'], 

                output_transform=jnn.softmax
            ),
        ),
        activation=ACT.tanh,
        activate_time=10,
        output_transform=ACT.sigmoid
    ),
    problem=problem,
    generation_limit=100,
    fitness_target=.8,
    seed=42,
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)
