from __future__ import print_function
from copy import copy, deepcopy
from heapq import heappush, heappop, heapify
from learning.pddlstream_utils import make_atom_map
import os

import time
import sys
import signal
from functools import partial, reduce
from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.advanced import enforce_simultaneous, identify_non_producers
from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.disabled import (
    push_disabled,
    reenable_disabled,
    process_stream_plan,
)
from pddlstream.algorithms.disable_skeleton import create_disabled_axioms

# from pddlstream.algorithms.downward import has_costs
from pddlstream.algorithms.incremental import process_stream_queue
from pddlstream.algorithms.instantiation import (
    GroundedInstance,
    HashableStreamResult,
    Instantiator,
    InformedInstantiator,
    ResultInstantiator,
    make_hashable,
)
from pddlstream.algorithms.refinement import (
    is_refined,
    iterative_plan_streams,
    get_optimistic_solve_fn,
    optimistic_stream_evaluation,
)
from pddlstream.algorithms.scheduling.plan_streams import OptSolution
from pddlstream.algorithms.reorder import reorder_stream_plan
from pddlstream.algorithms.skeleton import SkeletonQueue
from pddlstream.algorithms.visualization import (
    reset_visualizations,
    create_visualizations,
    has_pygraphviz,
    log_plans,
)
from pddlstream.language.constants import is_plan, get_length, str_from_plan, INFEASIBLE
from pddlstream.language.conversion import evaluation_from_fact
from pddlstream.language.fluent import compile_fluent_streams
from pddlstream.language.function import Function, Predicate
from pddlstream.language.object import SharedOptValue
from pddlstream.language.optimizer import ComponentStream
from pddlstream.algorithms.recover_optimizers import combine_optimizers
from pddlstream.language.statistics import (
    load_stream_statistics,
    write_stream_statistics,
    compute_plan_effort,
)
from pddlstream.language.stream import Stream, StreamInstance, StreamResult
from pddlstream.utils import HeapElement, INF, implies, str_from_object, safe_zip
from learning.visualization import visualize_atom_map


def get_negative_externals(externals):
    negative_predicates = list(
        filter(lambda s: type(s) is Predicate, externals)
    )  # and s.is_negative()
    negated_streams = list(
        filter(lambda s: isinstance(s, Stream) and s.is_negated, externals)
    )
    return negative_predicates + negated_streams


def partition_externals(externals, verbose=False):
    functions = list(filter(lambda s: type(s) is Function, externals))
    negative = get_negative_externals(externals)
    optimizers = list(
        filter(
            lambda s: isinstance(s, ComponentStream) and (s not in negative), externals
        )
    )
    streams = list(
        filter(lambda s: s not in (functions + negative + optimizers), externals)
    )
    if verbose:
        print(
            "Streams: {}\nFunctions: {}\nNegated: {}\nOptimizers: {}".format(
                streams, functions, negative, optimizers
            )
        )
    return streams, functions, negative, optimizers


##################################################


def recover_optimistic_outputs(stream_plan):
    if not is_plan(stream_plan):
        return stream_plan
    new_mapping = {}
    new_stream_plan = []
    for result in stream_plan:
        new_result = result.remap_inputs(new_mapping)
        new_stream_plan.append(new_result)
        if isinstance(new_result, StreamResult):
            opt_result = new_result.instance.opt_results[0]  # TODO: empty if disabled
            new_mapping.update(
                safe_zip(new_result.output_objects, opt_result.output_objects)
            )
    return new_stream_plan


def check_dominated(skeleton_queue, stream_plan):
    if not is_plan(stream_plan):
        return True
    for skeleton in skeleton_queue.skeletons:
        # TODO: has stream_plans and account for different output object values
        if frozenset(stream_plan) <= frozenset(skeleton.stream_plan):
            print(stream_plan)
            print(skeleton.stream_plan)
    raise NotImplementedError()


##################################################


def signal_handler(store, logpath, sig, frame):
    print("You pressed ctrl-C")
    if not (logpath is None):
        print(f"Logging statistics to {logpath + 'stats.json'}")
        store.write_to_json(logpath + "stats.json")
    sys.exit(0)


def solve_abstract(
    problem,
    constraints=PlanConstraints(),
    stream_info={},
    replan_actions=set(),
    unit_costs=False,
    success_cost=INF,
    max_time=INF,
    max_iterations=INF,
    max_memory=INF,
    initial_complexity=0,
    complexity_step=1,
    max_complexity=INF,
    max_skeletons=INF,
    search_sample_ratio=0,
    bind=True,
    max_failures=0,
    unit_efforts=False,
    max_effort=INF,
    effort_weight=None,
    reorder=True,
    visualize=False,
    verbose=True,
    logpath=None,
    oracle=None,
    use_unique=False,
    **search_kwargs,
):
    """
    Solves a PDDLStream problem by first planning with optimistic stream outputs and then querying streams
    :param problem: a PDDLStream problem
    :param constraints: PlanConstraints on the set of legal solutions
    :param stream_info: a dictionary from stream name to StreamInfo altering how individual streams are handled
    :param replan_actions: the actions declared to induce replanning for the purpose of deferred stream evaluation

    :param unit_costs: use unit action costs rather than numeric costs
    :param success_cost: the exclusive (strict) upper bound on plan cost to successfully terminate

    :param max_time: the maximum runtime
    :param max_iterations: the maximum number of search iterations
    :param max_memory: the maximum amount of memory

    :param initial_complexity: the initial stream complexity limit
    :param complexity_step: the increase in the stream complexity limit per iteration
    :param max_complexity: the maximum stream complexity limit

    :param max_skeletons: the maximum number of plan skeletons (max_skeletons=None indicates not adaptive)
    :param search_sample_ratio: the desired ratio of sample time / search time when max_skeletons!=None
    :param bind: if True, propagates parameter bindings when max_skeletons=None
    :param max_failures: the maximum number of stream failures before switching phases when max_skeletons=None

    :param unit_efforts: use unit stream efforts rather than estimated numeric efforts
    :param max_effort: the maximum amount of stream effort
    :param effort_weight: a multiplier for stream effort compared to action costs
    :param reorder: if True, reorder stream plans to minimize the expected sampling overhead

    :param visualize: if True, draw the constraint network and stream plan as a graphviz file
    :param verbose: if True, print the result of each stream application
    :param search_kwargs: keyword args for the search subroutine

    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan (INF if no plan), and evaluations is init expanded
        using stream applications
    """
    # TODO: select whether to search or sample based on expected success rates
    # TODO: no optimizers during search with relaxed_stream_plan
    # TODO: locally optimize only after a solution is identified
    # TODO: replan with a better search algorithm after feasible
    # TODO: change the search algorithm and unit costs based on the best cost
    use_skeletons = max_skeletons is not None
    # assert implies(use_skeletons, search_sample_ratio > 0)
    eager_disabled = effort_weight is None  # No point if no stream effort biasing
    num_iterations = eager_calls = 0
    complexity_limit = initial_complexity

    evaluations, goal_exp, domain, externals = parse_problem(
        problem,
        stream_info=stream_info,
        constraints=constraints,
        unit_costs=unit_costs,
        unit_efforts=unit_efforts,
        use_unique=use_unique,
    )
    identify_non_producers(externals)
    enforce_simultaneous(domain, externals)
    compile_fluent_streams(domain, externals)
    # TODO: make effort_weight be a function of the current cost
    # if (effort_weight is None) and not has_costs(domain):
    #     effort_weight = 1

    load_stream_statistics(externals)
    if visualize and not has_pygraphviz():
        visualize = False
        print("Warning, visualize=True requires pygraphviz. Setting visualize=False")
    if visualize:
        reset_visualizations()
    streams, functions, negative, optimizers = partition_externals(
        externals, verbose=verbose
    )
    eager_externals = list(filter(lambda e: e.info.eager, externals))
    positive_externals = streams + functions + optimizers
    has_optimizers = bool(optimizers)  # TODO: deprecate
    assert implies(has_optimizers, use_skeletons)

    ################

    store = SolutionStore(
        evaluations, max_time, success_cost, verbose, max_memory=max_memory
    )
    skeleton_queue = SkeletonQueue(store, domain, disable=not has_optimizers)
    disabled = set()  # Max skeletons after a solution

    signal.signal(signal.SIGINT, partial(signal_handler, store, logpath))
    if oracle is not None:
        oracle.set_infos(domain, externals, goal_exp, evaluations)

    while (
        (not store.is_terminated())
        and (num_iterations < max_iterations)
        and (complexity_limit <= max_complexity)
    ):
        num_iterations += 1
        eager_instantiator = Instantiator(
            eager_externals, evaluations
        )  # Only update after an increase?
        if eager_disabled:
            push_disabled(eager_instantiator, disabled)
        if eager_externals:
            eager_calls += process_stream_queue(
                eager_instantiator,
                store,
                complexity_limit=complexity_limit,
                verbose=verbose,
            )

        ###############

        print(
            "\nIteration: {} | Complexity: {} | Skeletons: {} | Skeleton Queue: {} | Disabled: {} | Evaluations: {} | "
            "Eager Calls: {} | Cost: {:.3f} | Search Time: {:.3f} | Sample Time: {:.3f} | Total Time: {:.3f}".format(
                num_iterations,
                complexity_limit,
                len(skeleton_queue.skeletons),
                len(skeleton_queue),
                len(disabled),
                len(evaluations),
                eager_calls,
                store.best_cost,
                store.search_time,
                store.sample_time,
                store.elapsed_time(),
            )
        )

        store.add_iteration_info(
            num_iterations,
            complexity_limit,
            len(skeleton_queue.skeletons),
            len(skeleton_queue),
            len(disabled),
            len(evaluations),
            eager_calls,
            store.best_cost,
            store.search_time,
            store.sample_time,
            store.elapsed_time(),
        )

        optimistic_solve_fn = get_optimistic_solve_fn(
            goal_exp,
            domain,
            negative,
            replan_actions=replan_actions,
            reachieve=use_skeletons,
            max_cost=min(store.best_cost, constraints.max_cost),
            max_effort=max_effort,
            effort_weight=effort_weight,
            **search_kwargs,
        )

        # TODO: just set unit effort for each stream beforehand
        if (max_skeletons is None) or (len(skeleton_queue.skeletons) < max_skeletons):
            disabled_axioms = (
                create_disabled_axioms(skeleton_queue) if has_optimizers else []
            )
            if disabled_axioms:
                domain.axioms.extend(disabled_axioms)
            stream_plan, opt_plan, cost = iterative_plan_streams(
                evaluations,
                positive_externals,
                optimistic_solve_fn,
                complexity_limit,
                store=store,
                max_effort=max_effort,
                oracle=oracle,
            )
            for axiom in disabled_axioms:
                domain.axioms.remove(axiom)
        else:
            stream_plan, opt_plan, cost = OptSolution(
                INFEASIBLE, INFEASIBLE, INF
            )  # TODO: apply elsewhere

        ################

        # stream_plan = replan_with_optimizers(evaluations, stream_plan, domain, externals) or stream_plan
        stream_plan = combine_optimizers(evaluations, stream_plan)
        # stream_plan = get_synthetic_stream_plan(stream_plan, # evaluations
        #                                       [s for s in synthesizers if not s.post_only])
        # stream_plan = recover_optimistic_outputs(stream_plan)
        if reorder:
            # TODO: this blows up memory wise for long stream plans
            stream_plan = reorder_stream_plan(store, stream_plan)

        num_optimistic = sum(r.optimistic for r in stream_plan) if stream_plan else 0
        action_plan = opt_plan.action_plan if is_plan(opt_plan) else opt_plan
        print(
            "Stream plan ({}, {}, {:.3f}): {}\nAction plan ({}, {:.3f}): {}".format(
                get_length(stream_plan),
                num_optimistic,
                compute_plan_effort(stream_plan),
                stream_plan,
                get_length(action_plan),
                cost,
                str_from_plan(action_plan),
            )
        )
        if is_plan(stream_plan) and visualize:
            log_plans(stream_plan, action_plan, num_iterations)
            create_visualizations(evaluations, stream_plan, num_iterations)

        ################

        if (
            (stream_plan is INFEASIBLE)
            and (not eager_instantiator)
            and (not skeleton_queue)
            and (not disabled)
        ):
            break
        if not is_plan(stream_plan):
            print(
                "No plan: increasing complexity from {} to {}".format(
                    complexity_limit, complexity_limit + complexity_step
                )
            )
            complexity_limit += complexity_step
            store.change_complexity(complexity_limit)
            if not eager_disabled:
                reenable_disabled(evaluations, domain, disabled)

        # print(stream_plan_complexity(evaluations, stream_plan))
        if not use_skeletons:
            process_stream_plan(
                store,
                domain,
                disabled,
                stream_plan,
                opt_plan,
                cost,
                bind=bind,
                max_failures=max_failures,
            )
            continue

        ################

        # optimizer_plan = replan_with_optimizers(evaluations, stream_plan, domain, optimizers)
        optimizer_plan = None
        if optimizer_plan is not None:
            # TODO: post process a bound plan
            print(
                "Optimizer plan ({}, {:.3f}): {}".format(
                    get_length(optimizer_plan),
                    compute_plan_effort(optimizer_plan),
                    optimizer_plan,
                )
            )
            skeleton_queue.new_skeleton(optimizer_plan, opt_plan, cost)

        allocated_sample_time = (
            (search_sample_ratio * store.search_time) - store.sample_time
            if len(skeleton_queue.skeletons) <= max_skeletons
            else INF
        )
        if (
            skeleton_queue.process(
                stream_plan, opt_plan, cost, complexity_limit, allocated_sample_time
            )
            is INFEASIBLE
        ):
            break

    ################

    summary = store.export_summary()
    summary.update(
        {
            "iterations": num_iterations,
            "complexity": complexity_limit,
            "skeletons": len(skeleton_queue.skeletons),
        }
    )

    store.change_complexity(complexity_limit)
    store.change_evaluations(summary["evaluations"])
    store.add_summary(summary)
    print(
        "Summary: {}".format(str_from_object(summary, ndigits=3))
    )  # TODO: return the summary

    write_stream_statistics(externals, verbose)
    if not (logpath is None):
        print(f"Logging statistics to {logpath + 'stats.json'}")
        store.write_to_json(logpath + "stats.json")

    return store.extract_solution()


solve_focused = solve_abstract  # TODO: deprecate solve_focused

##################################################


def solve_focused_original(problem, fail_fast=False, **kwargs):
    """
    Solves a PDDLStream problem by first planning with optimistic stream outputs and then querying streams
    :param problem: a PDDLStream problem
    :param fail_fast: whether to switch phases as soon as a stream fails
    :param kwargs: keyword args for solve_focused
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    max_failures = 0 if fail_fast else INF
    return solve_abstract(
        problem,
        max_skeletons=None,
        search_sample_ratio=None,
        bind=False,
        max_failures=max_failures,
        **kwargs,
    )


def solve_binding(problem, fail_fast=False, logpath=None, **kwargs):
    """
    Solves a PDDLStream problem by first planning with optimistic stream outputs and then querying streams
    :param problem: a PDDLStream problem
    :param fail_fast: whether to switch phases as soon as a stream fails
    :param kwargs: keyword args for solve_focused
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    max_failures = 0 if fail_fast else INF
    return solve_abstract(
        problem,
        max_skeletons=None,
        search_sample_ratio=None,
        bind=True,
        max_failures=max_failures,
        logpath=logpath,
        **kwargs,
    )


def solve_adaptive(
    problem,
    max_skeletons=INF,
    search_sample_ratio=1,
    logpath=None,
    oracle=None,
    use_unique=False,
    **kwargs,
):
    """
    Solves a PDDLStream problem by first planning with optimistic stream outputs and then querying streams
    :param problem: a PDDLStream problem
    :param max_skeletons: the maximum number of plan skeletons to consider
    :param search_sample_ratio: the desired ratio of search time / sample time
    :param kwargs: keyword args for solve_focused
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    max_skeletons = INF if max_skeletons is None else max_skeletons
    # search_sample_ratio = clip(search_sample_ratio, lower=0) # + EPSILON
    # assert search_sample_ratio > 0
    return solve_abstract(
        problem,
        max_skeletons=max_skeletons,
        search_sample_ratio=search_sample_ratio,
        bind=None,
        max_failures=None,
        logpath=logpath,
        oracle=oracle,
        use_unique=use_unique,
        **kwargs,
    )


def solve_hierarchical(problem, **kwargs):
    """
    Solves a PDDLStream problem by first planning with optimistic stream outputs and then querying streams
    :param problem: a PDDLStream problem
    :param search_sample_ratio: the desired ratio of sample time / search time
    :param kwargs: keyword args for solve_focused
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    return solve_adaptive(
        problem,
        max_skeletons=1,
        search_sample_ratio=INF,  # TODO: rename to sample_search_ratio
        bind=None,
        max_failures=None,
        **kwargs,
    )

class ResultQueue:

    def __init__(self):
        self.Q = []
        self.results = set()

    def push_result(self, new_result, score):
        assert isinstance(new_result, HashableStreamResult)
        heappush(
            self.Q,
            HeapElement(
                score,
                new_result
            ),
        )
        self.results.add(new_result)

    def pop_result(self):
        score, result = heappop(self.Q)
        self.results.remove(result)
        return score, result

    def __contains__(self, result):
        return result in self.results

    def __len__(self):
        return len(self.Q)

def should_planV2(iteration, Q):
    return (iteration % 10 == 0) or (len(Q) == 0)

class OptimisticResults:

    def __init__(self):
        self.results = set()
        self.list_results = []

    def add(self, result):
        assert isinstance(result, HashableStreamResult)
        if result in self.results:
            return
        # remove this soon
        assert not result.stream_fact in {r.stream_fact for r in self.results}
        self.results.add(result)
        self.list_results.append(result)

    def get_ordered_results(self, evaluations):
        result_set = set()
        # get a deduplicated version of self.list_results
        queue = []
        for i, result in list(enumerate(self.list_results))[::-1]: # reverse order to keep last added
            if result not in result_set:
                queue.insert(0, (i, result))
                result_set.add(result)

        evals = set(evaluations)
        ordered_inds = []
        ordered_results = []
        deferred = set()
        #TODO: is this correct?
        orphaned = set()
        while queue:
            (i, result) = queue.pop(0)
            if result not in self.results:
                continue
            domain = set(map(evaluation_from_fact, result.instance.get_domain()))
            if domain <= evals:
                ordered_results.append(result)
                ordered_inds.append(i)
                evals.update(map(evaluation_from_fact, result.get_certified()))
            else:
                if result in deferred:
                    orphaned.add(result)
                else:
                    queue.append((i, result))
                    deferred.add(result)

        self.list_results = [self.list_results[i] for i in ordered_inds] # update order
        return ordered_results, orphaned, evals

    def remove_result(self, result):
        if result.instance.external.is_negated:
            return
        if result.instance.external.is_fluent and result.instance.fluent_facts:
            matched = []
            for existing_result in self.results:
                if (result.instance.external is existing_result.instance.external and \
                    result.input_objects == existing_result.input_objects):
                    matched.append(existing_result)
            if not matched:
                print(result, 'not matched')
                return
            assert len(matched) == 1, "matched more than one result"
            result = matched[0]
        self.results.remove(result)
        return result

    #def remove_result_by_mapping:
    def remove_by_mapping(self, result, mapping):
        # update mapping of `result`
        result_c = make_hashable(result) #aka make a copy
        mapping = list(mapping.items())
        mapping_values = {v[1]:i for i, v in enumerate(mapping)}
        mapping_keys = [v[0] for v in mapping]
        for k,v in result_c.mapping.items():
            if v in mapping_values:
                result_c.mapping[k] = mapping_keys[mapping_values[v]]
        if result_c in self.results:
            return self.remove_result(result_c)
        return None
    
    def __len__(self):
        return len(self.results)

def reduce_score(score, num_visits, gamma = 0.8):
    return (gamma**num_visits)*score

def get_score_and_update_visits(instance_history, result, model, node_from_atom):
    # can't just use setdefault because it is not lazy
    instance = result.instance
    if instance in instance_history:
        score, num_visits = instance_history[instance]
        instance_history[instance] = (score, num_visits + 1)
    else:
        score = -model.predict(result, node_from_atom)
        num_visits = 0
        instance_history[instance] = (score, num_visits +1)
    return score, num_visits

def remove_orphaned(I_star, evaluations, instantiator):
    _, orphaned, evals = I_star.get_ordered_results(evaluations)
    for result in orphaned:
        removed = I_star.remove_result(result)
        if removed is not None: #TODO: is this nessecary
            instantiator.remove_certified_from_result(result)

def assert_no_orphans(I_star, evaluations):
    _, orphaned, evals = I_star.get_ordered_results(evaluations)
    assert not orphaned

def remove_disabled(I_star):
    for r in list(I_star.results): # is this a mega hack? What do do with disabled results?
        if r.instance.disabled:
            I_star.remove_result(r)

def solve_informedV2(
    problem,
    model,
    max_time=INF,
    max_iterations=INF,
    max_memory=INF,
    logpath=None,
    verbose=False,
    use_unique=True,
    search_sample_ratio=1,
    max_skeletons=INF,
    visualize_atom_maps=False,
    **search_kwargs,
):
    evaluations, goal_exp, domain, externals = parse_problem(
        problem,
        stream_info={},
        constraints=PlanConstraints(),
        unit_costs=False,
        unit_efforts=False,
        use_unique=use_unique,
    )
    # evaluations is a map from Evaluation(fact) to Node(complexity, result)
    identify_non_producers(externals)
    enforce_simultaneous(domain, externals)
    compile_fluent_streams(domain, externals)
    streams, functions, negative, optimizers = partition_externals(
        externals, verbose=verbose
    )
    optimistic_solve_fn = get_optimistic_solve_fn(
        goal_exp,
        domain,
        negative,
        max_cost=INF,
        max_effort=INF,
        effort_weight=None,
        **search_kwargs,
    )
    success_cost = INF
    store = SolutionStore(
        evaluations, max_time, success_cost, verbose, max_memory=max_memory
    )
    skeleton_queue = SkeletonQueue(store, domain, disable=True)
    signal.signal(signal.SIGINT, partial(signal_handler, store, logpath))
    model.set_infos(domain, externals, goal_exp, evaluations)
    iteration = 0
    instantiator = ResultInstantiator(streams)
    Q = ResultQueue()

    instance_history = {} # map from result to (original_score, num_visits)
    for opt_result in instantiator.initialize_results(evaluations):
        score = -model.predict(opt_result, instantiator.node_from_atom)
        Q.push_result(opt_result, score)
        instance_history[opt_result.instance] = (score, 1)

    I_star = OptimisticResults()
    while (
        (not store.is_terminated())
        and (iteration < max_iterations)
        and (instantiator or skeleton_queue.queue)
    ):
        iteration += 1
        force_sample = False
        if len(Q) > 0:
            score, result = Q.pop_result()
            # Need to check if this result has been orphaned since being placed on the queue
            # TODO: is this cheaper than cleaning the queue every time we remove results?
            assert_no_orphans(I_star, evaluations)
            _, _, all_facts = I_star.get_ordered_results(evaluations)
            if {evaluation_from_fact(f) for f in result.domain} <= all_facts: 
                # if result not in I_star.results: # do we want this here?
                # I dont think we want to filter here. See comment 1 below.
                # TODO: assert or check that result has not been expanded before
                new_results = instantiator.add_certified_from_result(result)
                if result.optimistic and not result.instance.disabled: # is this disabled thing correct?
                    I_star.add(result)
                for new_result in new_results:
                    if new_result in Q:
                        continue # do not re-add here
                    score, _ = get_score_and_update_visits(instance_history, new_result, model, instantiator.node_from_atom)
                    # TODO: should score be reduced here?
                    Q.push_result(new_result, score)
                assert_no_orphans(I_star, evaluations)
            else:
                pass
                # print(f"{result} orphaned since being added to queue")
        else:
            force_sample = True
            print("QUEUE EMPTY")

        if should_planV2(iteration, Q):
            ordered_results, _, _ = I_star.get_ordered_results(evaluations)
            print(
                f"Planning. # optim: {len(I_star)}. # grounded: {len(evaluations)}. Queue length: {len(Q)}"
            )
            if visualize_atom_maps:
                visualize_atom_map(
                    make_atom_map(instantiator.node_from_atom), os.path.join(logpath, f"atom_map_{iteration}.html")
                )
            stream_plan, opt_plan, cost = optimistic_solve_fn(
                evaluations, ordered_results, None, store=store
            )  # psi, pi*
            if is_plan(opt_plan) and not is_refined(stream_plan):
                print("Found Unrefined Plan")
                new_results, bindings = optimistic_stream_evaluation(
                    evaluations, stream_plan
                )  # \bar{psi}, B
                bound_objects = set(bindings)
                for result in stream_plan:
                    if (
                        result.optimistic
                        and set(result.output_objects) <= bound_objects
                    ):
                        # there is a reason not to do this.
                        # an example would be: find-motion(#o3, #o4) -> #t1
                        # where #o3 and #o4 are the unrefined outputs of all the IK's

                        # but wouldn't #o3 and #o4 become refined in the process above?
                        # no i think it would only get the one meaning of #o3 #o4 used in the eval

                        # so how do we know when to remove and when to keep?
                        # first guess: if there exists a result that is in I* but not stream_plan
                        # which produces any of the input_objects of result
                        opt_inputs = set(o for o in result.input_objects if isinstance(o.value, SharedOptValue))
                        if opt_inputs:
                            remove = True
                            for other_result in ordered_results:
                                if other_result not in stream_plan:
                                    other_opt_outputs = set(o for o in other_result.output_objects if isinstance(o.value, SharedOptValue))
                                    has_other_producers = (other_opt_outputs & opt_inputs)
                                    if has_other_producers:
                                        remove= False
                                        break
                            if not remove:
                                continue


                        # i know this design is slower, but I am making this explicit to start
                        removed = I_star.remove_result(result)
                        if removed is not None: #TODO: is this nessecary
                            instantiator.remove_certified_from_result(result)

                remove_orphaned(I_star, evaluations, instantiator)
                assert_no_orphans(I_star, evaluations)

                for result in new_results:
                    result = make_hashable(result)
                    if (result.instance.external.is_negated) or (result in I_star.results):
                        continue

                    # TODO: add on queue or instantiate everything?
                    # comment 1: These results MUST be popped off the queue in the order that they appear in new_results.
                    # We need to first add the results to I_star, then put it on the queue for expansion
                    I_star.add(result)
                    if result  not in Q:
                        score, _ = get_score_and_update_visits(
                            instance_history, result, model, instantiator.node_from_atom
                        )
                        Q.push_result(result, score)
                
                assert_no_orphans(I_star, evaluations)
                
                continue
            elif is_plan(opt_plan):
                print_refined_plan(opt_plan)
                force_sample = True
            else:
                force_sample = False

        else:
            stream_plan = None

        if force_sample:# or iteration % 100 == 0:
            allocated_sample_time = (
                (search_sample_ratio * store.search_time) - store.sample_time
                if len(skeleton_queue.skeletons) <= max_skeletons
                else INF
            )
            #allocated_sample_time = max(allocated_sample_time, 5)
            skeleton_queue.process(
                stream_plan, opt_plan, cost, 0, allocated_sample_time
            )
            # add new grounded results, push them on the queue for later expansion
            for result in skeleton_queue.new_results:
                result = make_hashable(result)
                if result.instance.external.is_negated or result.instance.fluent_facts: # what is this doing?
                    continue
                # TODO: check if grounded instance will be treated the same as optimistic instance (ie same hash)
                score, num_visits = get_score_and_update_visits(
                    instance_history, result, model, instantiator.node_from_atom
                )
                score = reduce_score(score, num_visits) 
                instantiator.add_certified_from_result(result, force_add = False, expand = False)
                assert not result.optimistic
                assert result not in Q
                Q.push_result(result, score)

            for processed_result, mapping in skeleton_queue.processed_results:
                result = I_star.remove_by_mapping(processed_result, mapping)
                if result is not None:
                    instantiator.remove_certified_from_result(result)
                    # TODO: check if this disabled thing is correct (what does it even mean?)
                    if not (result.instance.enumerated or result.instance.disabled):
                        score, num_visits = get_score_and_update_visits(
                            instance_history, result, model, instantiator.node_from_atom
                        )
                        score = reduce_score(score, num_visits) 
                        Q.push_result(result, score)
            

            remove_disabled(I_star)
            remove_orphaned(I_star, evaluations, instantiator)
            assert_no_orphans(I_star, evaluations)

            # add new grounded results, push them on the queue for later expansion

    ################

    summary = store.export_summary()
    summary.update(
        {
            "iterations": iteration,
            "complexity": INF,
            "skeletons": len(skeleton_queue.skeletons),
        }
    )

    store.change_complexity(INF)
    store.change_evaluations(summary["evaluations"])
    store.add_summary(summary)
    print(
        "Summary: {}".format(str_from_object(summary, ndigits=3))
    )  # TODO: return the summary

    write_stream_statistics(externals, verbose)
    if not (logpath is None):
        print(f"Logging statistics to {logpath + 'stats.json'}")
        store.write_to_json(os.path.join(logpath, "stats.json"))

    return store.extract_solution()

def print_refined_plan(opt_plan):   
    print("Found a refined plan!")
    print("Length", len(opt_plan.action_plan))
    prev_name = ""
    wrong = False
    for action in opt_plan.action_plan:
        if action.name == prev_name:
            wrong = True
        print("\t", action.name, [o.pddl for o in action.args])
        prev_name = action.name

    if wrong:
        print("This plan is NON-OPTIMAL")

# TODO: the atom map is not being modified by the instantiator
def solve_informed(
    problem,
    model,
    max_time=INF,
    max_iterations=INF,
    max_memory=INF,
    logpath=None,
    verbose=False,
    use_unique=True,
    search_sample_ratio=1,
    max_skeletons=INF,
    visualize_atom_maps=False,
    **search_kwargs,
):
    evaluations, goal_exp, domain, externals = parse_problem(
        problem,
        stream_info={},
        constraints=PlanConstraints(),
        unit_costs=False,
        unit_efforts=False,
        use_unique=use_unique,
    )

    identify_non_producers(externals)
    enforce_simultaneous(domain, externals)
    compile_fluent_streams(domain, externals)
    # load_stream_statistics(externals)
    streams, functions, negative, optimizers = partition_externals(
        externals, verbose=verbose
    )
    positive_externals = streams + functions + optimizers

    optimistic_solve_fn = get_optimistic_solve_fn(
        goal_exp,
        domain,
        negative,
        max_cost=INF,
        max_effort=INF,
        effort_weight=None,
        **search_kwargs,
    )
    success_cost = INF
    store = SolutionStore(
        evaluations, max_time, success_cost, verbose, max_memory=max_memory
    )
    skeleton_queue = SkeletonQueue(store, domain, disable=True)
    signal.signal(signal.SIGINT, partial(signal_handler, store, logpath))
    model.set_infos(domain, externals, goal_exp, evaluations)
    num_iterations = 0
    instantiator = InformedInstantiator(streams, evaluations, model)  # Q
    iteration = 0
    visits = []
    while (
        (not store.is_terminated())
        and (num_iterations < max_iterations)
        and (instantiator or skeleton_queue.queue)
    ):
        iteration += 1
        if instantiator:
            priority, instance = instantiator.pop_instance()
            visits.append(instance)
            if isinstance(instance, GroundedInstance):
                result, complexity = instance.result, instance.complexity
                for fact in result.get_certified():
                    atom = evaluation_from_fact(fact)
                    instantiator.add_atom(atom, complexity, create_instances=True)
            else:
                if instance.enumerated:
                    continue
                for i, result in enumerate(instance.next_optimistic()):
                    assert i == 0, "Why?!"
                    complexity = instantiator.compute_complexity(instance)
                    instantiator.add_result(result, complexity, create_instances=True)
                    # TODO: should there not be a push_instance here to push all the new instances on the queue
        else:
            print("Queue empty!")
        # if instance.is_refined():
        # num_visits = priority.num_visits + 1
        #     # Never want two of the same unrefined fact in the planning problem
        #     # and the only way for the original optimistic fact to have been removed
        #     # is if we refined the instance, in which case we already have a new refined instance
        #     score = reduce_score(priority.score, num_visits)
        #     instantiator.push_instance(instance, score=score, num_visits=num_visits)

        if should_plan(
            iteration, priority.score, instantiator.optimistic_results, instantiator
        ):
            print(
                f"Planning with {len(instantiator.optimistic_results)} results and {len(evaluations)} evaluations. Queue size: {len(instantiator.queue)}"
            )
            stream_plan, opt_plan, cost = optimistic_solve_fn(
                evaluations, instantiator.ordered_results, None, store=store
            )  # psi, pi*
            if is_plan(opt_plan) and not is_refined(stream_plan):
                print("Found UNrefined plan!")
                # refine stuff
                new_results, bindings = optimistic_stream_evaluation(
                    evaluations, stream_plan
                )  # \bar{psi}, B
                bound_objects = set(bindings)

                for result in stream_plan:
                    if (
                        result.optimistic
                        and set(result.output_objects) <= bound_objects
                    ):
                        instantiator.remove_result(result)

                for result in new_results:
                    if result.instance.external.is_negated:
                        continue
                    complexity = instantiator.compute_complexity(
                        result.instance
                    )  # TODO: is this right?
                    instantiator.add_result(
                        result, complexity, create_instances=True
                    )  # This is necessary to preserve the order of dependent results
                    # TODO: maybe defer adding these results by adding to queue?
                # TODO: check crappiness
                continue

            elif is_plan(opt_plan):
                print("Found a refined plan!")
                print("Length", len(opt_plan.action_plan))
                prev_name = ""
                wrong = False
                for action in opt_plan.action_plan:
                    if action.name == prev_name:
                        wrong = True
                        print("Someting is wrong")
                    print("\t", action.name, [o.pddl for o in action.args])
                    prev_name = action.name

                if wrong:
                    print("Something is wrong with this optimistic plan")

                force_sample = True
            else:
                # not is_plan
                force_sample = False
        else:
            force_sample = False
            stream_plan = None

        if force_sample or should_sample(iteration, skeleton_queue):
            # allocated_sample_time = (
            #     (search_sample_ratio * store.search_time) - store.sample_time
            #     if len(skeleton_queue.skeletons) <= max_skeletons
            #     else INF
            # )
            allocated_sample_time = 5
            skeleton_queue.process(
                stream_plan, opt_plan, cost, 0, allocated_sample_time
            )

            for result in skeleton_queue.new_results:
                if result.instance.external.is_negated or result.instance.fluent_facts:
                    continue
                instance = result.instance
                complexity = result.compute_complexity(store.evaluations)
                instantiator.record_complexity(result, complexity)
                ground_instance = GroundedInstance(instance, result, complexity)
                instantiator.push_grounded_instance(grounded_instance=ground_instance)

            # remove all old results that have been processed
            grounded_instances = {r.instance for r in skeleton_queue.new_results}
            enumerated, grounded = 0, 0
            for result in list(instantiator.optimistic_results):
                # newly enumerated ones
                if result.instance.enumerated:
                    if result.optimistic:
                        enumerated += 1
                        instantiator.remove_result(result)

                # newly grounded ones.
                # TODO: Do we want to remove all the optimistic results that share an instance with a newly grounded result?
                elif result.instance in grounded_instances:
                    grounded += 1
                    # I belive this line below will not remove the results it should:
                    instantiator.remove_result(result)
                    # TODO: the line below seems to cause the same plan to be found repeatedly
                    instantiator.push_or_reduce_score(result.instance)
            
            # we need to do something here to add all grounded results at once (so the atom map is correct)

            print(
                f"Removed {grounded} grounded and {enumerated} enumerated optimistic results"
            )

    ################

    summary = store.export_summary()
    summary.update(
        {
            "iterations": num_iterations,
            "complexity": INF,
            "skeletons": len(skeleton_queue.skeletons),
        }
    )

    store.change_complexity(INF)
    store.change_evaluations(summary["evaluations"])
    store.add_summary(summary)
    print(
        "Summary: {}".format(str_from_object(summary, ndigits=3))
    )  # TODO: return the summary

    write_stream_statistics(externals, verbose)
    if not (logpath is None):
        print(f"Logging statistics to {logpath + 'stats.json'}")
        store.write_to_json(os.path.join(logpath, "stats.json"))

    return store.extract_solution()


def should_sample(iteration, skeleton_queue):
    return False
    # return bool(skeleton_queue.queue) and iteration % 100 == 0


last_seen = -1


def should_plan(iteration, score, results, instantiator):
    # return True
    # global last_seen
    # if score > last_seen:
    #    last_seen = score
    #    return True
    # else:
    #    return not bool(len(instantiator.queue))
    return len(results) % 10 == 0 or not instantiator
