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
        self.evaluations = evaluations
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
        self.evaluations = evaluations
        self.initialize_atom_map(evaluations)
        self.optimistic_results = set()
        self.instance_history = {}
        self.__list_results = []
        for stream in self.streams:
            if not stream.domain:
                assert not stream.inputs
                self.push_instance(stream.get_instance([]))

        for atom, node in evaluations.items():
            self.add_atom(atom, node.complexity)


    def add_result(self, result, complexity, create_instances=True):
        assert result not in self.optimistic_results, "Why?!"
        if result.stream_fact in {r.stream_fact for r in self.optimistic_results}:
            return
        for fact in result.get_certified():
            self.add_atom(evaluation_from_fact(fact), complexity, create_instances=create_instances)

        self.optimistic_results.add(result)
        self.__list_results.append(result)


    def record_complexity(self, result, complexity):
        for fact in result.get_certified():
            self.add_atom(evaluation_from_fact(fact), complexity, create_instances=False)

    @property
    def ordered_results(self):
        queue = list(enumerate(self.__list_results))
        evals = set(self.evaluations)
        ordered_inds = []
        ordered_results = []
        deferred = set()
        orphaned = []
        while queue:
            (i, result) = queue.pop(0)
            if result not in self.optimistic_results:
                continue
            domain = set(map(evaluation_from_fact, result.instance.get_domain()))
            if domain <= evals:
                ordered_results.append(result)
                ordered_inds.append(i)
                evals.update(map(evaluation_from_fact, result.get_certified()))
            else:
                # assert result not in deferred, "Found an orphaned stream result"
                if result in deferred:
                    # print("Removed orphaned result", result)
                    orphaned.append(result)
                else:
                    queue.append((i, result))
                    deferred.add(result)

        self.__list_results = [self.__list_results[i] for i in ordered_inds] # update order
        for result in orphaned:
            self.remove_result(result)
        print(f'Removed {len(orphaned)} orphaned results')
        return ordered_results

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
                print(result, 'not matched')
                return
            result = matched[0]
        self.optimistic_results.remove(result)

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
        # if head in self.complexity_from_atom:
        #     assert self.complexity_from_atom[head] <= complexity
        # else:
        self.complexity_from_atom[head] = complexity
        if create_instances:
            self._add_new_instances(head)

    def remove_atom(self, atom):
        if not is_atom(atom):
            return False
        head = atom.head
        del self.complexity_from_atom[head]
        



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
