from __future__ import print_function

import time
import sys
import signal
from functools import partial
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
from pddlstream.algorithms.instantiation import GroundedInstance, Instantiator, InformedInstantiator
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
from pddlstream.language.optimizer import ComponentStream
from pddlstream.algorithms.recover_optimizers import combine_optimizers
from pddlstream.language.statistics import (
    load_stream_statistics,
    write_stream_statistics,
    compute_plan_effort,
)
from pddlstream.language.stream import Stream, StreamInstance, StreamResult
from pddlstream.utils import INF, implies, str_from_object, safe_zip


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
    use_unique = False,
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
        use_unique = use_unique,
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
    use_unique = False,
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
        use_unique = use_unique,
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
    ** search_kwargs
):
    evaluations, goal_exp, domain, externals = parse_problem(
        problem,
        stream_info={},
        constraints=PlanConstraints(),
        unit_costs=False,
        unit_efforts=False,
        use_unique = use_unique
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
    skeleton_queue = SkeletonQueue(store, domain, disable=False)
    signal.signal(signal.SIGINT, partial(signal_handler, store, logpath))
    model.set_infos(domain, externals, goal_exp, evaluations)
    num_iterations = 0
    instantiator = InformedInstantiator(streams, evaluations, model) # Q
    while (
        (not store.is_terminated())
        and (num_iterations < max_iterations)
        and (instantiator or skeleton_queue.queue)
    ):
        if instantiator:
            priority, instance = instantiator.pop_instance()
            if isinstance(instance, GroundedInstance):
                instantiator.add_result(instance.result, instance.complexity)
            else:
                if instance.enumerated:
                    continue
                for i, result in enumerate(instance.next_optimistic()):
                    assert i == 0, "Why?!"
                    complexity = instantiator.compute_complexity(instance)
                    instantiator.add_result(result, complexity)
        else:
            print("Queue empty!")
        # if instance.is_refined():
            # num_visits = priority.num_visits + 1
        #     # Never want two of the same unrefined fact in the planning problem
        #     # and the only way for the original optimistic fact to have been removed
        #     # is if we refined the instance, in which case we already have a new refined instance
        #     score = reduce_score(priority.score, num_visits)
        #     instantiator.push_instance(instance, score=score, num_visits=num_visits)

        if should_plan(priority.score, instantiator.optimistic_results, instantiator):
            stream_plan, opt_plan, cost = optimistic_solve_fn(evaluations, instantiator.ordered_results, None, store=store) # psi, pi*
            if is_plan(opt_plan) and not is_refined(stream_plan):
                # refine stuff
                new_results, bindings = optimistic_stream_evaluation(evaluations, stream_plan) # \bar{psi}, B
                bound_objects = set(bindings)

                for result in stream_plan:
                    if set(result.output_objects) <= bound_objects:
                        instantiator.remove_result(result)

                instantiator.remove_orphans()

                for result in new_results:
                    if result.instance.external.is_negated:
                        continue
                    instantiator.push_instance(result.instance, readd=True)
                    # TODO: remove this forloop when we no longer need ComplexityModel
                    complexity = instantiator.compute_complexity(result.instance)
                    instantiator.add_result(result, complexity, create_instances=False)
                
                # TODO: check crappiness
                continue
            
            elif is_plan(opt_plan):
                print("Found a refined plan!")
                force_sample = True
            else:
                # not is_plan
                force_sample = False
        else:
            force_sample = False
            stream_plan = None

        if force_sample or should_sample(skeleton_queue):
            allocated_sample_time = (
                (search_sample_ratio * store.search_time) - store.sample_time
                if len(skeleton_queue.skeletons) <= max_skeletons
                else INF
            )
            skeleton_queue.process(
                stream_plan, opt_plan, cost, 0, allocated_sample_time
            )
            
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
                elif result.optimistic and result.instance in grounded_instances:
                    grounded += 1
                    instantiator.remove_result(result)
            print(f'Removed {grounded} grounded and {enumerated} enumerated optimistic results')
            instantiator.remove_orphans()

            for result in skeleton_queue.new_results:
                if result.instance.external.is_negated or result.instance.fluent_facts:
                    continue
                instance = result.instance
                complexity = result.compute_complexity(store.evaluations)
                ground_instance = GroundedInstance(instance, result, complexity)
                instantiator.push_grounded_instance(grounded_instance=ground_instance)
                instantiator.add_result(result, complexity, create_instances=False)

            for instance in grounded_instances:
                if not instance.enumerated:
                    instantiator.push_or_reduce_score(instance)


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
        store.write_to_json(logpath + "stats.json")

    return store.extract_solution()


def should_sample(skeleton_queue):
    return bool(skeleton_queue.queue)

def should_plan(score, results, instantiator):
    return True
    return len(results) % 10 == 0
