from collections import defaultdict, namedtuple, Sized
from heapq import heappush, heappop, heapify
from itertools import product, count
import random

from pddlstream.algorithms.common import COMPLEXITY_OP
from pddlstream.algorithms.relation import compute_order, Relation, solve_satisfaction
from pddlstream.language.constants import is_parameter
from pddlstream.language.conversion import evaluation_from_fact, fact_from_evaluation, is_atom, head_from_fact
from pddlstream.utils import safe_zip, HeapElement, safe_apply_mapping

USE_RELATION = True

# TODO: maybe store unit complexity here as well as a tiebreaker
Priority = namedtuple('Priority', ['complexity', 'num']) # num ensures FIFO

def is_instance(atom, schema):
    return (atom.function == schema.function) and \
            all(is_parameter(b) or (a == b)
                for a, b in safe_zip(atom.args, schema.args))

def test_mapping(atoms1, atoms2):
    mapping = {}
    for a1, a2 in safe_zip(atoms1, atoms2):
        assert a1.function == a2.function
        for arg1, arg2 in safe_zip(a1.args, a2.args):
            if mapping.get(arg1, arg2) == arg2:
                mapping[arg1] = arg2
            else:
                return None
    return mapping

##################################################

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.7049&rep=rep1&type=pdf

class Instantiator(Sized): # Dynamic Instantiator
    def __init__(self, streams, evaluations={}, verbose=False):
        # TODO: lazily instantiate upon demand
        self.streams = streams
        self.verbose = verbose
        #self.streams_from_atom = defaultdict(list)
        self.queue = []
        self.num_pushes = 0 # shared between the queues
        # TODO: rename atom to head in most places
        self.complexity_from_atom = {}
        self.atoms_from_domain = defaultdict(list)
        for stream in self.streams:
            if not stream.domain:
                assert not stream.inputs
                self.push_instance(stream.get_instance([]))
        for atom, node in evaluations.items():
            self.add_atom(atom, node.complexity)
        # TODO: revisit deque and add functions to front
        # TODO: record the stream instances or results?

    #########################

    def __len__(self):
        return len(self.queue)

    def compute_complexity(self, instance):
        domain_complexity = COMPLEXITY_OP([self.complexity_from_atom[head_from_fact(f)]
                                           for f in instance.get_domain()] + [0])
        return domain_complexity + instance.external.get_complexity(instance.num_calls)

    def push_instance(self, instance):
        # TODO: flush stale priorities?
        complexity = self.compute_complexity(instance)
        priority = Priority(complexity, self.num_pushes)
        heappush(self.queue, HeapElement(priority, instance))
        self.num_pushes += 1
        if self.verbose:
            print(self.num_pushes, instance)

    def pop_stream(self):
        priority, instance = heappop(self.queue)
        return instance

    def min_complexity(self):
        priority, _ = self.queue[0]
        return priority.complexity

    #########################

    def _add_combinations(self, stream, atoms):
        if not all(atoms):
            return
        domain = list(map(head_from_fact, stream.domain))
        # Most constrained variable/atom to least constrained
        for combo in product(*atoms):
            mapping = test_mapping(domain, combo)
            if mapping is not None:
                input_objects = safe_apply_mapping(stream.inputs, mapping)
                self.push_instance(stream.get_instance(input_objects))

    def _add_combinations_relation(self, stream, atoms):
        if not all(atoms):
            return
        # TODO: might be a bug here?
        domain = list(map(head_from_fact, stream.domain))
        # TODO: compute this first?
        relations = [Relation(filter(is_parameter, domain[index].args),
                              [tuple(a for a, b in safe_zip(atom.args, domain[index].args)
                                     if is_parameter(b)) for atom in atoms[index]])
                     for index in compute_order(domain, atoms)]
        solution = solve_satisfaction(relations)
        for element in solution.body:
            mapping = solution.get_mapping(element)
            input_objects = safe_apply_mapping(stream.inputs, mapping)
            self.push_instance(stream.get_instance(input_objects))

    def _add_new_instances(self, new_atom):
        for s_idx, stream in enumerate(self.streams):
            for d_idx, domain_fact in enumerate(stream.domain):
                domain_atom = head_from_fact(domain_fact)
                if is_instance(new_atom, domain_atom):
                    # TODO: handle domain constants more intelligently
                    self.atoms_from_domain[s_idx, d_idx].append(new_atom)
                    atoms = [self.atoms_from_domain[s_idx, d2_idx] if d_idx != d2_idx else [new_atom]
                              for d2_idx in range(len(stream.domain))]
                    if USE_RELATION:
                        self._add_combinations_relation(stream, atoms)
                    else:
                        self._add_combinations(stream, atoms)

    def add_atom(self, atom, complexity):
        if not is_atom(atom):
            return False
        head = atom.head
        if head in self.complexity_from_atom:
            assert self.complexity_from_atom[head] <= complexity
            return False
        self.complexity_from_atom[head] = complexity
        self._add_new_instances(head)
        return True

        
ModelPriority = namedtuple('Priority', ['score']) # num ensures FIFO
class InformedInstantiator(Instantiator):
    def __init__(self, streams, evaluations, model, verbose=False):
        self.streams = streams
        self.verbose = verbose
        self.model = model

        self.queue = []
        self.num_pushes = 0 # shared between the queues

        self.complexity_from_atom = {}
        self.atoms_from_domain = defaultdict(list)
        self.initialize_atom_map(evaluations)
        self.optimistic_results = set()
        self.output_object_to_results = {}
        self.instance_history = {}
        self.__list_results = []
        for stream in self.streams:
            if not stream.domain:
                assert not stream.inputs
                self.push_instance(stream.get_instance([]))

        for atom, node in evaluations.items():
            self.add_atom(atom, node.complexity)
            for output_object in atom.head.args:
                self.output_object_to_results[output_object] = None

    def add_result(self, result, complexity, create_instances=True):
        assert result not in self.optimistic_results, "Why?!"
        assert result.stream_fact not in {r.stream_fact for r in self.optimistic_results}
        for fact in result.get_certified():
            self.add_atom(evaluation_from_fact(fact), complexity, create_instances=create_instances)

        if create_instances:
            self.optimistic_results.add(result)
            self.__list_results.append(result)
            for o in result.output_objects:
                self.output_object_to_results.setdefault(o, set()).add(result)
    @property
    def ordered_results(self):
        remove_inds = []
        ordered_list = []
        for i, result in enumerate(self.__list_results):
            if result not in self.optimistic_results:
                remove_inds.append(i)
            else:
                ordered_list.append(result)
        for index in remove_inds[::-1]:
            self.__list_results.pop(index)
        return ordered_list
    def remove_result(self, result):
        if result.instance.external.is_negated:
            return
        if result.instance.external.is_fluent and result.instance.fluent_facts:
            # find the one that matches (barring fluent_facts)
            matched = []
            for existing_result in self.optimistic_results:
                if (result.instance.external is existing_result.instance.external and \
                    result.input_objects == existing_result.input_objects):
                    matched.append(existing_result)
            if not matched:
                return
            result = matched[0]
        self.optimistic_results.remove(result)
        for o in result.output_objects:
            self.output_object_to_results[o].remove(result)
        # TODO: remove corresponding atoms

    def initialize_atom_map(self, evaluations):
        self.atom_map = {fact_from_evaluation(f): [] for f, _ in evaluations.items()}

    def pop_instance(self):
        priority, instance = heappop(self.queue)
        return priority, instance

    def reduce_score(self, score, num_visits, decay_factor=0.8):
        return score * (decay_factor**num_visits)

    def find_element(self, instance):
        # check if instance already on the queue
        for heap_element in self.queue:
            if heap_element.value is instance:
                return heap_element
        return None

    def push_grounded_instance(self, grounded_instance, new_visits=1):
        assert self.find_element(grounded_instance) is None
        if grounded_instance.instance in self.instance_history:
            score, num_visits = self.instance_history[grounded_instance.instance]
        else:
            score = self.model.predict(grounded_instance.instance, atom_map=self.atom_map, instantiator=self)
            num_visits = 0
        self.instance_history[grounded_instance.instance] = (score, num_visits + new_visits)
        priority = ModelPriority(self.reduce_score(score, num_visits + new_visits))
        heappush(self.queue, HeapElement(priority, grounded_instance))
        self.num_pushes += 1

    def push_or_reduce_score(self, instance):
        heap_element = self.find_element(instance)
        if heap_element is None:
            return self.push_instance(instance, readd=True)

        new_score = self.score_instance(instance)
        heap_element.key = ModelPriority(new_score)
        heapify(self.queue)

    def score_instance(self, instance):
        if instance in self.instance_history:
            score, num_visits = self.instance_history[instance]
            self.instance_history[instance] = (score, num_visits + 1)
            score = self.reduce_score(score, num_visits + 1)
        else:
            score = self.model.predict(instance, atom_map=self.atom_map, instantiator=self)
            self.instance_history[instance] = (score, 0)

        return score

    def push_instance(self, instance, readd=False):
        if not readd and instance in self.instance_history:
            """This is here because when we refine a result, we'll add its instance, without calling add_atom
            on its parent facts. But then, later, when we pop its parent result from the queue, and compute
            the newly possible instances, the same instance might show up again. So now readd has to be explicit."""
            return
        elif instance in self.instance_history:
            assert readd

        score = self.score_instance(instance)
        priority = ModelPriority(score)
        heappush(self.queue, HeapElement(priority, instance))
        self.num_pushes += 1
        if self.verbose:
            print(self.num_pushes, instance)

    def add_atom(self, atom, complexity, create_instances=True):
        if not is_atom(atom):
            return False
        head = atom.head
        if head in self.complexity_from_atom:
            assert self.complexity_from_atom[head] <= complexity
        else:
            self.complexity_from_atom[head] = complexity
        if create_instances:
            self._add_new_instances(head)

    def remove_atom(self, atom):
        if not is_atom(atom):
            return False
        head = atom.head
        del self.complexity_from_atom[head]
        
    def is_orphan(self, obj, cache={}):
        if obj in cache:
            return cache[obj]
        results = self.output_object_to_results[obj]
        if results is None:
            res = False
        elif len(results) == 0:
            res = True
        else:
            res = any(
                any(self.is_orphan(in_obj) for in_obj in result.input_objects) \
                    for result in results
            )
        cache[obj] = res
        return res

    def remove_orphans(self):
        cache = {}
        to_remove = []
        for stream_result in self.optimistic_results:
            if any(
                self.is_orphan(obj, cache) \
                    for obj in stream_result.input_objects
                ):
                to_remove.append(stream_result)
        for stream_result in to_remove:
            self.remove_result(stream_result)

        # TODO: maybe make this faster? Could just check on pop of queue.
        to_remove = []
        for i in range(len(self.queue)):
            instance = self.queue[i].value
            if isinstance(instance, GroundedInstance):
                continue
            if any(
                self.is_orphan(obj, cache) \
                    for obj in instance.input_objects
                ):
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.queue.pop(i)
        print(f'Removed {len(to_remove)} orphans')
        heapify(self.queue)

class GroundedInstance:
    def __init__(self, instance, result, complexity):
        self.instance = instance
        self.result = result
        self.complexity = complexity
    def __hash__(self):
        return hash((self.instance, self.result))
    def __eq__(self, other):
        return isinstance(other, GroundedInstance) and other.instance is self.instance and other.result is self.result



def should_sample(skeleton_queue):
    return bool(skeleton_queue.queue)

def should_plan(score, results, instantiator):
    return True
    return len(results) % 10 == 0
