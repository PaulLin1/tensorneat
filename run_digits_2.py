from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem
from tensorneat.genome import DefaultGenome, BiasNode

problem = DigitsClassificationProblem()
# input_coors = [
#     (x / 3.5 - 1.0, y / 3.5 - 1.0)
#     for y in range(8)
#     for x in range(8)
# ]

input_coors = [
    (x / 3.5 - 1.0, y / 3.5 - 1.0)
    for y in range(8)
    for x in range(8)
]
input_coors.append((0.0, -1.2))  # bias node

hidden_coors = [(x / 9 * 2 - 1, y / 9 * 2 - 1) for y in range(10) for x in range(10)]

output_coors = [(x / 9 * 2 - 1, 1) for x in range(10)]

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors
)
pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
    ),
    problem=problem,
    generation_limit=300,
    fitness_target=1,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)
