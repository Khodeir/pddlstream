from __future__ import print_function

from collections import namedtuple
from copy import deepcopy

from pddlstream.algorithms.common import add_fact, INTERNAL_EVALUATION
from pddlstream.algorithms.downward import make_predicate, make_preconditions, make_effects, add_predicate
from pddlstream.language.constants import Or, And, is_parameter, Equal, Not, str_from_plan, EQ
from pddlstream.language.object import Object, OptimisticObject
from pddlstream.utils import find_unique, safe_zip, str_from_object, INF, is_hashable, incoming_from_edges

OrderedSkeleton = namedtuple('OrderedSkeleton', ['actions', 'orders']) # TODO: AND/OR tree

WILD = '*'
ASSIGNED_PREDICATE = '{}assigned'
GROUP_PREDICATE = '{}group'
ORDER_PREDICATE = '{}order'

GOAL_INDEX = -1

def linear_order(actions):
    if not actions:
        return set()
    return {(i, i+1) for i in range(len(actions)-1)} \
           | {(len(actions)-1, GOAL_INDEX)}

class PlanConstraints(object):
    def __init__(self, skeletons=None, groups={}, exact=True, hint=False, max_cost=INF):
        if skeletons is not None:
            skeletons = [skeleton if isinstance(skeleton, OrderedSkeleton)
                         else OrderedSkeleton(skeleton, linear_order(skeleton)) for skeleton in skeletons]
        self.skeletons = skeletons
        self.groups = groups # Could make this a list of lists
        self.exact = exact
        self.max_cost = max_cost
        #self.max_length = max_length
        #self.hint = hint # Search over skeletons first and then fall back
        #if self.hint:
        #    raise NotImplementedError()
    def dump(self):
        print('{}(exact={}, max_cost={})'.format(self.__class__.__name__, self.exact, self.max_cost))
        if self.skeletons is None:
            return
        for i, skeleton in enumerate(self.skeletons):
            print(i, str_from_plan(skeleton))
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, str_from_object(self.__dict__))

# TODO: rename other costs to be terminate_cost (or decision cost)

def to_constant(parameter):
    name = parameter[1:]
    return to_obj('@{}'.format(name))

def to_obj(value):
    # Allows both raw values as well as objects to be specified
    if any(isinstance(value, Class) for Class in [Object, OptimisticObject]):
        return value
    return Object.from_value(value)

##################################################

def make_assignment_facts(assigned_predicate, local_from_global, parameters):
    return [(assigned_predicate, to_constant(p), local_from_global[p])
            for p in parameters]

def make_order_facts(order_predicate, num, steps):
    return [(order_predicate, to_obj('n{}'.format(num)), to_obj('t{}'.format(step)))
            for step in range(steps)]

def get_internal_prefix(internal):
    return '_' if internal else ''

def add_plan_constraints(constraints, domain, evaluations, goal_exp, internal=False):
    if (constraints is None) or (constraints.skeletons is None):
        return goal_exp
    import pddl
    # TODO: can search over skeletons first and then fall back
    # TODO: unify this with the constraint ordering
    # TODO: can constrain to use a plan prefix
    prefix = get_internal_prefix(internal)
    assigned_predicate = ASSIGNED_PREDICATE.format(prefix)
    group_predicate = GROUP_PREDICATE.format(prefix)
    order_predicate = ORDER_PREDICATE.format(prefix)
    new_facts = []
    for group in constraints.groups:
        for value in constraints.groups[group]:
            # TODO: could make all constants groups (like an equality group)
            fact = (group_predicate, to_obj(group), to_obj(value))
            new_facts.append(fact)
    new_actions = []
    new_goals = []
    for num, skeleton in enumerate(constraints.skeletons):
        actions, orders = skeleton
        incoming_orders = incoming_from_edges(orders)
        order_facts = make_order_facts(order_predicate, num, len(actions))
        bound_parameters = set()
        for step, (name, args) in enumerate(actions):
            # TODO: could also just remove the free parameter from the action
            new_action = deepcopy(find_unique(lambda a: a.name == name, domain.actions))
            constant_pairs = [(a, p.name) for a, p in safe_zip(args, new_action.parameters)
                         if not is_parameter(a) and a != WILD]
            skeleton_parameters = list(filter(is_parameter, args))
            existing_parameters = [p for p in skeleton_parameters if p in bound_parameters]
            new_parameters = [p for p in skeleton_parameters if p not in bound_parameters]
            local_from_global = {a: p.name for a, p in safe_zip(args, new_action.parameters) if is_parameter(a)}

            # TODO: either not assigned or ensure the same
            group_preconditions = [(group_predicate if is_hashable(a) and (a in constraints.groups) else EQ,
                                    to_obj(a), p) for a, p in constant_pairs]
            new_preconditions = make_assignment_facts(assigned_predicate, local_from_global, existing_parameters) + \
                                group_preconditions + [order_facts[idx] for idx in incoming_orders[step]]
            new_action.precondition = pddl.Conjunction(
                [new_action.precondition, make_preconditions(new_preconditions)]).simplified()

            new_effects = make_assignment_facts(assigned_predicate, local_from_global, new_parameters) \
                          + [order_facts[step]] # + [Not(order_facts[step-1])]
            new_action.effects.extend(make_effects(new_effects))
            # TODO: should also negate the effects of all other sequences here

            new_actions.append(new_action)
            bound_parameters.update(new_parameters)
            #new_action.dump()
        new_goals.append(And(*[order_facts[idx] for idx in incoming_orders[GOAL_INDEX]]))
    add_predicate(domain, make_predicate(order_predicate, ['?num', '?step']))
    if constraints.exact:
        domain.actions[:] = []
    domain.actions.extend(new_actions)
    new_goal_exp = And(goal_exp, Or(*new_goals))
    for fact in new_facts:
        add_fact(evaluations, fact, result=INTERNAL_EVALUATION)
    return new_goal_exp
