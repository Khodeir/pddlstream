from __future__ import print_function
from copy import copy, deepcopy
from heapq import heappush, heappop
from itertools import count
from learning.pddlstream_utils import fact_to_pddl, make_atom_map
import os

import time
import sys
import signal
from functools import partial, reduce
from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.advanced import enforce_simultaneous, identify_non_producers
from pddlstream.algorithms.common import SolutionStore, add_certified
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
from pddlstream.algorithms.scheduling.recover_streams import Node
from pddlstream.algorithms.skeleton import SkeletonQueue
from pddlstream.algorithms.visualization import (
    reset_visualizations,
    create_visualizations,
    has_pygraphviz,
    log_plans,
)
from pddlstream.language.constants import is_plan, get_length, str_from_plan, INFEASIBLE
from pddlstream.language.conversion import evaluation_from_fact, fact_from_evaluation
from pddlstream.language.fluent import compile_fluent_streams
from pddlstream.language.function import Function, Predicate
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

def signal_handler(store, logpath, sig, frame, extra_info = None):
    print("You pressed ctrl-C")
    summary = store.export_summary()

    if extra_info is not None:
        print(extra_info)
        summary.update(
            {
                "iterations": extra_info["num_iterations"],
                "complexity": extra_info["complexity_limit"],
                "skeletons": extra_info["num_skeletons"],
            }
        )

        store.change_complexity(extra_info["complexity_limit"])
    store.change_evaluations(summary["evaluations"])
    store.add_summary(summary)
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
    problem_file_path = None,
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
    extra_info = {} #TODO: fix this crap
    extra_info["num_skeletons"] = 0
    extra_info["complexity_limit"] = 0
    extra_info["num_iterations"] = 0
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
        evaluations, max_time, success_cost, verbose, max_memory=max_memory, problem_file_path = problem_file_path
    )
    skeleton_queue = SkeletonQueue(store, domain, disable=not has_optimizers)
    disabled = set()  # Max skeletons after a solution

    signal.signal(signal.SIGINT, partial(signal_handler,  store, logpath, extra_info = extra_info))
    if oracle is not None:
        oracle.set_infos(domain, externals, goal_exp, evaluations)

    while (
        (not store.is_terminated())
        and (num_iterations < max_iterations)
        and (complexity_limit <= max_complexity)
    ):
        num_iterations += 1
        extra_info["num_iterations"] = num_iterations
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

        extra_info["num_skeletons"] = len(skeleton_queue.skeletons)
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
            extra_info["complexity_limit"] = complexity_limit
            store.change_complexity(complexity_limit)
            if not eager_disabled:
                reenable_disabled(evaluations, domain, disabled)

        # print(stream_plan_complexity(evaluations, stream_plan))
        if is_plan(opt_plan) and is_plan(stream_plan):
            print_pddl_plan(opt_plan)
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
        self.count = count()

    def push_result(self, new_result, score):
        assert isinstance(new_result, HashableStreamResult)
        assert new_result not in self.results
        heappush(
            self.Q,
            HeapElement(
                (score, next(self.count)),
                new_result
            ),
        )
        self.results.add(new_result)

    def pop_result(self):
        (score, _), result = heappop(self.Q)
        self.results.remove(result)
        return score, result

    def __contains__(self, result):
        return result in self.results

    def __len__(self):
        return len(self.Q)

def should_planV2(iteration, Q, reachable, N=10, K=20):
    if not hasattr(should_planV2, 'last_reachable_count'):
        should_planV2.last_reachable_count = 0
    res = False
    if len(Q) == 0:
        res = True
    elif (iteration % K == 0) and (len(reachable) >= should_planV2.last_reachable_count + N):
        res = True

    if res:
        should_planV2.last_reachable_count = len(reachable)
    
    return res

class OptimisticResults:

    def __init__(self):
        self.results = set()
        self.ordered_results = list()
        self.reachable_evals = set()
        self.level = dict()
        self.node_from_atom = dict()
        self.unrefined_facts = set()
        self.atom_map = {}

    def add(self, result):
        assert isinstance(result, HashableStreamResult)
        assert self.supports(result)
        if result in self.results:
            return
        self.results.add(result)
        # TODO: revisit this. because why would there ever be duplicates?
        assert result not in self.ordered_results
        self.ordered_results.append(result)
        self.add_facts(result)
        
    def supports(self, result):
        return {evaluation_from_fact(f) for f in result.domain} <= self.reachable_evals
    
    def add_facts(self, result):
        domain = {evaluation_from_fact(f) for f in result.domain}
        l = max(self.level[f] for f in domain)
        is_refined = result.is_refined() and all(f not in self.unrefined_facts for f in domain)
        for atom in result.get_certified():
            cert = evaluation_from_fact(atom)
            self.level[cert] = min(l + 1, self.level.get(cert, l + 1))
            self.reachable_evals.add(cert)

            if is_refined:
                assert result.is_refined() and result.is_input_refined_recursive()
                self.node_from_atom[atom] = self.node_from_atom.get(atom, Node(0, result))
                self.atom_map[fact_to_pddl(atom)] = [fact_to_pddl(f) for f in result.domain]
            else:
                assert not (result.is_refined() and result.is_input_refined_recursive())
                self.unrefined_facts.add(cert)
                

    def update_reachable(self, evaluations, assert_no_orphans=False):
        '''
        Orders the reuslts in order of precedence
        Computes the total set of reachable facts
        Computes the "level" of all reachable facts
        Removes unsupported (orphaned) results
        '''
        result_set = set()
        # get a deduplicated version of self.ordered_results
        queue = []
        for result in list(self.ordered_results)[::-1]: # reverse order to keep last added
            if result not in result_set:
                queue.insert(0, result)
                result_set.add(result)

        evals = set(evaluations)
        node_from_atom = {}
        atom_map = {}
        for atom in evaluations:
            fact = fact_from_evaluation(atom)
            node_from_atom[fact] = Node(0, evaluations[atom].result)
            if not isinstance(evaluations[atom].result, bool):
                atom_map[fact_to_pddl(fact)] = [fact_to_pddl(f) for f in evaluations[atom].result.domain] if (evaluations[atom].result is not None) else []
        ordered_results = []
        deferred = set()
        orphaned = set()
        level = {e:node.complexity for e, node in evaluations.items()}
        unrefined_facts = set()
        while queue:
            result = queue.pop(0)
            if result not in self.results:
                continue
            domain = set(map(evaluation_from_fact, result.instance.get_domain()))
            is_refined = result.is_refined() and all(f not in self.unrefined_facts for f in domain)
            if domain <= evals:
                ordered_results.append(result)
                l = max(level[f] for f in domain)
                for atom in result.get_certified():
                    cert = evaluation_from_fact(atom)
                    level[cert] = min(l + 1, level.get(cert, l + 1))
                    evals.add(cert)
                    if is_refined:
                        assert result.is_refined() and result.is_input_refined_recursive()
                        node_from_atom[atom] = node_from_atom.get(atom, Node(0, result))
                        atom_map[fact_to_pddl(atom)] = [fact_to_pddl(f) for f in result.domain]
                    else:
                        assert not (result.is_refined() and result.is_input_refined_recursive())
                        unrefined_facts.add(cert)
            else:
                if result in deferred:
                    orphaned.add(result)
                else:
                    queue.append(result)
                    deferred.add(result)

        self.ordered_results = ordered_results
        self.reachable_evals = evals
        self.level = level
        self.node_from_atom = node_from_atom
        self.atom_map = atom_map

        if assert_no_orphans:
            assert not orphaned 
        else:
            for result in orphaned:
                removed = self.remove_result(result)

        assert set(ordered_results) == self.results

    def remove_result(self, result):
        assert not result.instance.external.is_negated
        assert not (result.external.is_fluent and result.instance.fluent_facts)
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

    def remove_disabled(self):
        for r in list(self.results):
            if r.instance.disabled:
                self.remove_result(r)
    
    def __len__(self):
        return len(self.results)

def reduce_score(score, num_visits, gamma = 0.8):
    return (gamma**num_visits)*score

def get_score_and_update_visits(instance_history, result, model, node_from_atom, levels, store, atom_map = None):
    # can't just use setdefault because it is not lazy
    start_time = time.time()
    instance = result.instance
    is_refined = instance.is_refined()
    if instance in instance_history:
        score, num_visits, was_refined = instance_history[instance]
        if was_refined == is_refined:
            instance_history[instance] = (score, num_visits + 1, was_refined)
            store.record_scoring(time.time() - start_time)
            return score, num_visits
        else:
            assert not was_refined and is_refined, "Somehow the instance got unrefined"
            # TODO: going to reset visits to 0. Right?
            # print("Instance has been refined since last time. Rescoring.", instance)
    num_visits = 0
    score = -model.predict(result, node_from_atom, levels=levels, atom_map = atom_map)
    instance_history[instance] = (score, num_visits +1, is_refined)
    store.record_scoring(time.time() - start_time)
    return score, num_visits

def remove_orphaned(I_star, evaluations, instantiator):
    I_star.update_reachable(evaluations)
    instantiator.update_reachable_evaluations(I_star.reachable_evals)

def assert_no_orphans(I_star, evaluations):
    assert I_star.update_reachable(evaluations, assert_no_orphans=True) is None

def get_instance_without_fluents(result):
    assert result.external.is_fluent and result.instance.fluent_facts
    return result.external.get_instance(result.input_objects)

def get_opt_result_no_fluents(result):
    instance = get_instance_without_fluents(result)
    if instance.opt_results:
        assert len(instance.opt_results) == 1
        opt = instance.opt_results[0]
        return make_hashable(opt)
    return None

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
    stream_info={},
    eager_mode=False,
    problem_file_path = None,
    **search_kwargs,
):
    evaluations, goal_exp, domain, externals = parse_problem(
        problem,
        stream_info=stream_info,
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
    model.set_infos(domain, externals, goal_exp, evaluations)
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
        evaluations, max_time, success_cost, verbose, max_memory=max_memory, problem_file_path = problem_file_path
    )
    skeleton_queue = SkeletonQueue(store, domain, disable=True)
    signal.signal(signal.SIGINT, partial(signal_handler, store, logpath))
    iteration = 0
    last_sample_time = time.time()
    instantiator = ResultInstantiator(streams)
    Q = ResultQueue()
    I_star = OptimisticResults()
    I_star.update_reachable(evaluations, assert_no_orphans=True)
    ALLOW_CHILDREN_BEFORE_EXPANSION = False # whether to add results to instantiator prior to expansion
    EAGER_MODE = eager_mode # whether to add all new results to I_star immediately
    expanded = set()
    instance_history = {} # map from result to (original_score, num_visits)
    for opt_result in instantiator.initialize_results(evaluations):
        score, _ = get_score_and_update_visits(instance_history, opt_result, model, node_from_atom=I_star.node_from_atom, levels=I_star.level, store=store, atom_map = I_star.atom_map)
        if EAGER_MODE:
            I_star.add(opt_result)
        if ALLOW_CHILDREN_BEFORE_EXPANSION:
            instantiator.add_certified_from_result(opt_result, expand=False)
        if opt_result not in Q:
            Q.push_result(opt_result, score)
    I_star.update_reachable(evaluations, assert_no_orphans=True)

    while (
        (not store.is_terminated())
        and (iteration < max_iterations)
        and (instantiator or skeleton_queue.queue)
    ):
        iteration += 1
        force_sample = False
        if len(Q) > 0:
            score, result = Q.pop_result()
            assert all(head_set <= {e.head for e in I_star.reachable_evals} for head_set in instantiator.atoms_from_domain.values())
            if result.optimistic and result.instance.disabled:
                # TODO: We get here because we removed disabled from I_star, but not queue
                assert result not in I_star.results
                continue
            elif result.optimistic:
                if EAGER_MODE:
                    if result not in I_star.results:
                        continue # been removed
                    assert result in I_star.results
                    assert I_star.supports(result)
                else:
                    if not I_star.supports(result):
                        continue # been orphaned
                    I_star.add(result)
            else:
                assert all(evaluation_from_fact(f) in evaluations for f in result.get_certified())

            assert result not in expanded, result
            new_results = instantiator.add_certified_from_result(result, expand=True)
            expanded.add(result)

            for new_result in new_results:
                assert all(head_set <= {e.head for e in I_star.reachable_evals} for head_set in instantiator.atoms_from_domain.values())
                if EAGER_MODE:
                    I_star.add(new_result)
                if ALLOW_CHILDREN_BEFORE_EXPANSION:
                    instantiator.add_certified_from_result(new_result, expand=False)
                if new_result in Q or new_result in expanded:
                    # the only way this happens is if these results were added as part of
                    # refinement or sampling 
                    continue # do not re-add here
                score, _ = get_score_and_update_visits(instance_history, new_result, model, I_star.node_from_atom, I_star.level, store=store, atom_map = I_star.atom_map)
                # TODO: should score be reduced here?
                Q.push_result(new_result, score)
            assert_no_orphans(I_star, evaluations)
        else:
            force_sample = True
            print("QUEUE EMPTY")

        if should_planV2(iteration, Q, I_star.reachable_evals):
            store.change_results(len(I_star.results))
            print(
                f"Planning. # optim: {len(I_star)}. # grounded: {len(evaluations)}. Queue length: {len(Q)}. Expanded: {len(expanded)}.", end=" "
            )
            if visualize_atom_maps:
                visualize_atom_map(
                    make_atom_map(I_star.node_from_atom), os.path.join(logpath, f"atom_map_{iteration}.html")
                )
            start_plan = time.time()
            stream_plan, opt_plan, cost = optimistic_solve_fn(
                evaluations, I_star.ordered_results, None, store=store
            )  # psi, pi*
            print(f'duration: {time.time() - start_plan}')
            if is_plan(opt_plan) and not is_refined(stream_plan):
                print("Found Unrefined Plan")
                # print_pddl_plan(opt_plan)
                new_results, bindings = optimistic_stream_evaluation(
                    evaluations, stream_plan
                )  # \bar{psi}, B
                bound_objects = set(bindings)
                stream_plan = [get_opt_result_no_fluents(r) if r.instance.fluent_facts else make_hashable(r) for r in stream_plan]
                assert all(isinstance(r, HashableStreamResult) for r in stream_plan)
                new_results = [get_opt_result_no_fluents(r) if r.instance.fluent_facts else make_hashable(r) for r in new_results]
                assert all(isinstance(r, HashableStreamResult) for r in new_results)
                for result in stream_plan:
                    if (
                        result.optimistic
                        and result in I_star.results # if result is negative it wont appear in I_star
                        and result not in new_results # could be already refined
                    ):
                        # the optimistic version of the fluent results have different output objects from
                        # the one used in the stream_plan e.g. #t50 vs #t20 so the subset check doesnt work.
                        if not (set(result.output_objects) <= bound_objects or result.external.is_fluent):
                            continue
                        assert (
                            result.external.is_fluent # will be added in new_results
                            or (not result.is_refined())
                            or any(o.is_shared() for o in result.input_objects)
                        ), (result, bindings)
                        # We dont want to remove the result if its inputs
                        # can be produced by other stream results not part of this plan
                        # an example would be: find-motion(#o3, #o4) -> #t1
                        # where #o3 and #o4 are the unrefined outputs of all the IK's
                        opt_inputs = set(o for o in result.input_objects if o.is_shared())
                        if opt_inputs:
                            remove = True
                            for other_result in I_star.ordered_results:
                                if other_result not in stream_plan:
                                    other_opt_outputs = set(o for o in other_result.output_objects if o.is_shared())
                                    has_other_producers = (other_opt_outputs & opt_inputs)
                                    if has_other_producers:
                                        remove = False
                                        break
                            if not remove:
                                continue
                        removed = I_star.remove_result(result)

                for result in new_results:
                    if (result.instance.external.is_negated) or (not result.optimistic):
                        continue

                    if result.instance.fluent_facts:
                        # I'm curious about when this would ever happen
                        assert False, f"Newly refined result {result} has fluent facts"
                        continue

                    # comment 1: We need to first add the results to I_star, then put it on the queue for expansion
                    I_star.add(result)
                    if ALLOW_CHILDREN_BEFORE_EXPANSION:
                        instantiator.add_certified_from_result(result, expand=False)

                    if result not in Q and result not in expanded:
                        score, _ = get_score_and_update_visits(
                            instance_history, result, model, I_star.node_from_atom, I_star.level, store=store, atom_map = I_star.atom_map
                        )
                        Q.push_result(result, score)

                remove_orphaned(I_star, evaluations, instantiator)         
                assert_no_orphans(I_star, evaluations)
                
                continue
            elif is_plan(opt_plan):
                print('Found refined plan')
                print_pddl_plan(opt_plan)
                force_sample = True
            else:
                force_sample = False

        else:
            stream_plan = None

        since_last_sample = (time.time() - last_sample_time)
        if (not force_sample) and (len(Q) == 0) and len(skeleton_queue.queue) == 0:
            break
        elif force_sample or len(skeleton_queue.skeletons) > max_skeletons or len(Q) == 0 or (skeleton_queue.skeletons and since_last_sample > 5):
            if force_sample:
                sample_time = max(search_sample_ratio * since_last_sample, 2)
            else:
                sample_time = search_sample_ratio * since_last_sample
            skeleton_queue.process(
                stream_plan, opt_plan, cost, 0, sample_time
            )
            last_sample_time = time.time()
            I_star.update_reachable(evaluations, assert_no_orphans=True)
            # add new grounded results, push them on the queue for later expansion
            for result in skeleton_queue.new_results:
                result = make_hashable(result)
                if result.instance.external.is_negated or result.instance.fluent_facts: # what is this doing?
                    continue
                assert not result.optimistic
                if ALLOW_CHILDREN_BEFORE_EXPANSION:
                    # Add certified so that the newly grounded facts are added to node from atom, and can be
                    # used to instantiate new results in subsequent iterations
                    instantiator.add_certified_from_result(result, expand = False)
                # TODO: check if grounded instance will be treated the same as optimistic instance (ie same hash)
                score, num_visits = get_score_and_update_visits(
                    instance_history, result, model, I_star.node_from_atom, I_star.level, store=store, atom_map = I_star.atom_map
                )
                score = reduce_score(score, num_visits)

                assert result not in Q
                assert result not in expanded
                Q.push_result(result, score)

            for processed_result, mapping in skeleton_queue.processed_results:
                if processed_result.instance.external.is_negated:
                    continue
                if processed_result.instance.fluent_facts:
                    processed_result = get_opt_result_no_fluents(processed_result)
                    if processed_result is None:
                        continue

                assert processed_result.external.is_fluent or processed_result.instance.disabled, processed_result
                result = I_star.remove_by_mapping(processed_result, mapping)     
                # TODO: I need to convince myself that grounded facts from fluent streams
                # can be reused and "remapped" with new fluent facts. Otherwise, we might
                # need to add them back
                # if result is not None:
                #     assert result.external.is_fluent or result.instance.disabled
                #     # TODO: If this assert never fires we can get rid of the things below
                #     if not (result.instance.enumerated or result.instance.disabled):
                #         score, num_visits = get_score_and_update_visits(
                #             instance_history, result, model, I_star.node_from_atom
                #         )
                #         score = reduce_score(score, num_visits) 
                #         Q.push_result(result, score)
            
            I_star.remove_disabled()
            remove_orphaned(I_star, evaluations, instantiator)
            assert_no_orphans(I_star, evaluations)
            store.change_evaluations(len(evaluations))

    ################

    summary = store.export_summary()
    summary.update(
        {
            "iterations": iteration,
            "complexity": INF,
            "skeletons": len(skeleton_queue.skeletons),
        }
    )
    summary['expanded'] = len(expanded)

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

    model.after_run(store=store, expanded=expanded, logpath=logpath)
    return store.extract_solution()

def print_pddl_plan(opt_plan):   
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
