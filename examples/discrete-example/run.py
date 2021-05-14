import os
import numpy as np
from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
ARRAY = tuple

domain_pddl = '''(define (domain example)
    (:requirements :strips)
    (:predicates
        (location ?l)
        (atconf ?l)
        (empty)
    )
    (:action move-free
        :parameters (?start ?end)
        :precondition (and (location ?start) (location ?end) (empty) (atconf ?start))
        :effect (and (not (atconf ?start)) (atconf ?end))
    )    
)'''
stream_pddl = '''(define (stream example)
  (:stream sample-location
    :outputs (?l)
    :certified (and (location ?l))
  )
)'''
constant_map = {}

start_loc = ARRAY([1, 1])
init = [('empty',), ('location', start_loc), ('atconf', start_loc)]
goal = ('atconf', ARRAY([0, 1]))

def get_location(size=3):
    locs = []
    for i in range(size):
        for j in range(size):
            locs.append((i,j))
    np.random.shuffle(locs)
    while locs:
        yield (ARRAY(locs.pop()), )

stream_map = {
    'sample-location': from_gen_fn(get_location)
}

problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

print('Initial:', str_from_object(problem.init))
print('Goal:', str_from_object(problem.goal))
for algorithm in ['adaptive', 'binding', 'focused', 'incremental']:
    solution = solve(problem, algorithm=algorithm, verbose=False)
    print(f"\n\n{algorithm} solution:")
    print_solution(solution)
    input('Continue')



