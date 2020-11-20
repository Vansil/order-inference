import numpy as np
import evaluation.ground_truth

import pickle
from operator import add
from functools import reduce
import os
from sys import getsizeof
from collections import defaultdict
import itertools
import random
from copy import copy

from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt

import networkx as nx


class Solver(object):
    '''
    Evolution Strategy
    '''
    
    def __init__(self, evaluator, nvars=None):
        self.evaluator = evaluator
        if nvars is None:
            self.n_vars = evaluator.n_vars
        else:
            self.n_vars = nvars

        self.population_size = 100
        self.offspring_size = 400
        self.elite_size = 5
        
        self.recomb_rate = 1
        self.recomb_mode = 'cyclic'
        
        self.inversion_size = self.mutation_size_per_relation_swap()
        self.mutation_inversion_rate = 0
        self.mutation_swap_rate = 0
        
    def mutation_size_per_relation_swap(self):
        """
        number of consecutive variables to mirror in order to swap 
        on average one ground-truth relation (assuming square gt matrix)
            n(n-1)/2 = 1/x (n: #vars, x: percent true)
        """
        return int(np.ceil(
            .5 + .5 * np.sqrt(1 + 8/self.evaluator.percentage_true())
        ))
        
    def run(self,verbose=False):
        results = []
        # initialisation
        population = self.initialise_population()
        population_fitness = self.evaluate(population)
        
        if verbose:
            print("generation\tbest fitness\tdiversity")
        
        generation_count = 0
        best_fitnesses = []
        for _ in range(200):
            # parent selection
            mating_pool = self.select_parents(population)

            # recombination
            offspring = self.recombination(mating_pool)

            # mutation
            self.mutation(offspring) # in-place

            offspring_fitness = self.evaluate(offspring)

            # survivor selection
            population, population_fitness = self.select_survivors(
                population, population_fitness, 
                offspring, offspring_fitness)
            
            # print
            diversity = self.estimate_diversity(population)
            if (verbose and generation_count % 1 == 0):
                print("{:03d}\t\t{:.4f}\t\t{:.2f}".format(
                    generation_count, 
                    max(population_fitness),
                    diversity
                ))
            generation_count += 1

            # add to results
            results.append((
                generation_count,
                max(population_fitness),
                diversity,
                population[np.argmax(population_fitness)]
            ))

            # early stopping
            best_fitness = max(population_fitness)
            best_fitnesses.append(best_fitness)
            if best_fitness == self.evaluator.total_true():
                print("Stopped: maximum fitness reached")
                break
            if len(best_fitnesses)>30 and best_fitnesses[-1]==best_fitnesses[-30]:
                print("Stopped: no improvement for 30 generations")
                break
            
        return results
    
    def initialise_population(self):
        return [np.random.permutation(self.n_vars) for _ in range(self.population_size)]
            
    def evaluate(self, population):
        return list(map(self.evaluator.evaluate, population))
    
    def select_parents(self, population):
        # Uniform random selection
        matingPool = []
        for _ in range(int(self.offspring_size/2)):
            matingPool.append(random.sample(population,2))
        return matingPool
    
    def recombination(self, mating_pool):
        if self.recomb_mode == 'cyclic':
            return self.recombination_cyclic(mating_pool)
        else:
            return self.recombination_direct(mating_pool)
    
    def recombination_direct(self, mating_pool):
        """
        Cross over one random section, fill out the remainder from left to right
        """
        offspring = []
        for parent1, parent2 in mating_pool:
            if np.random.random() > self.recomb_rate:
                offspring += [parent1, parent2]
            else:
                indices = sorted(np.random.choice(range(self.n_vars),2,replace=False))
                swap1 = parent1[indices[0]:indices[1]]
                swap2 = parent2[indices[0]:indices[1]]
                remainder1 = [x for x in parent1 if x not in swap2]
                remainder2 = [x for x in parent2 if x not in swap1]
                offspring.append(np.hstack((
                    remainder1[:indices[0]],
                    swap2,
                    remainder1[indices[0]:]
                )).astype(np.int))
                offspring.append(np.hstack((
                    remainder2[:indices[0]],
                    swap1,
                    remainder2[indices[0]:]
                )).astype(np.int))
        return offspring
    
    def recombination_cyclic(self, mating_pool):
        """
        Cyclic cross-over
        """
        offspring = []
        for parent1, parent2 in mating_pool:
            if np.random.random() > self.recomb_rate:
                offspring += [parent1, parent2]
            else:
                child1 = copy(parent1)
                child2 = copy(parent2)
                indices_todo = set(range(len(parent1)))
                do_swap = False # alternate between swapping indices and not swapping
                while indices_todo:
                    # get cycle starting from the first index that hasn't been in a cycle yet
                    cycle = []
                    new_index = list(indices_todo)[0]
                    while not cycle or not new_index == cycle[0]:
                        cycle.append(new_index)
                        indices_todo.remove(new_index)
                        value = parent2[cycle[-1]]
                        new_index = np.where(parent1==value)[0][0]
                    # swap values
                    do_swap = not do_swap
                    if do_swap:
                        child1[cycle] = parent2[cycle]
                        child2[cycle] = parent1[cycle]
                offspring += [child1, child2]
        return offspring
    
    def mutation(self, offspring):
        self.mutation_swap(offspring)
        self.mutation_inversion(offspring)
    
    def mutation_swap(self, offspring):
        """
        RESULTS: not very effective, much computation time
        Mirror random indices with probability of *mutation_swap_rate*
        """
        n = self.n_vars
        p = self.mutation_swap_rate
        for individual in offspring:
            # compute number of swaps
            nSwaps = max(0,int(random.gauss(p*n, np.sqrt(n*p*(1-p)))))
            for _ in range(nSwaps):
                indices = np.random.choice(range(n), 2, replace=False)
                # mirror values
                individual[indices] = individual[list(reversed(indices))]
        
    
    def mutation_inversion(self, offspring):
        """
        RESULTS: not very effective
        Mirror *mutation_size* adjacent indices with probability of *mutation_inversion_rate*
        """
        n = self.n_vars
        p = self.mutation_inversion_rate
        s = self.inversion_size
        for individual in offspring:
            # compute number of swaps
            nSwaps = max(0,int(random.gauss(p*n, np.sqrt(n*p*(1-p)))))
            for index in np.random.choice(n-s+1, nSwaps):
                # mirror values
                individual[index:index+s-1] = list(reversed(individual[index:index+s-1]))
                   
    def select_survivors(self, population, population_fitness, offspring, offspring_fitness):
        # get elite
        elite_indices = np.argsort(population_fitness)
        elite = [population[i] for i in elite_indices[-self.elite_size:]]
        elite_fitness = [population_fitness[i] for i in elite_indices[-self.elite_size:]]
        # combine offspring and elite
        pool = offspring + elite
        pool_fitness = offspring_fitness + elite_fitness
        # select best
        best_indices = np.argsort(pool_fitness)
        best = [pool[i] for i in best_indices[-self.population_size:]]
        best_fitness = [pool_fitness[i] for i in best_indices[-self.population_size:]]
        return best, best_fitness
    
    def estimate_diversity(self, population):
        # using Hamming distance
        N = 100
        n_ind = len(population)
        hamming_sum = 0
        for _ in range(N):
            i1, i2 = np.random.choice(range(n_ind), 2, replace=False)
            hamming_sum += sum(population[i1]!=population[i2])
        return hamming_sum/N




class EvaluatorBinary(object):
    """
    Takes a binary ground-truth based on an absolute threshold and
    counts how many true relations are satisfied by the order
    """
    def __init__(self, data, threshold=0.7):
        self.evaluationCount = 0
        _, data_int, self.int_pos = data
        self.true_desc = self.find_true_descendants(abs(data_int) > threshold)
        self.n_vars, self.n_inters = data_int.shape
    
    def check_phenotype(self, phenotype):
        assert sorted(phenotype)==list(range(self.n_vars)), 'Incorrect phenotype format'
        
    def percentage_true(self):
        # return percentage of true relations
        return sum(map(len,self.true_desc)) / self.n_vars / self.n_inters
    
    def total_true(self):
        return sum(map(len,self.true_desc))
        
    def find_true_descendants(self, gt_matrix):
        """
        Returns per intervention index a list of true descendants
        """
        return [set(np.nonzero(gt_matrix[:,inter])[0]) for inter in range(gt_matrix.shape[1])]
    
    def evaluate(self, phenotype):
        self.check_phenotype(phenotype)
        self.evaluationCount += 1
        fitness = 0
        # loop over interventions
        for inter_id, inter_index in enumerate(self.int_pos):
            # select allowed descendants in phenotype
            pheno_desc = phenotype[list(phenotype).index(inter_index)+1:]
            # count correct by taking intersection with actual descendants
            fitness += len(self.true_desc[inter_id].intersection(pheno_desc))
        return fitness


class EvaluatorContinuous(object):
    """
    Adds all intervention values that violate the order
    """
    def __init__(self, data):
        from order.evaluate import Evaluator
        self.evaluator = Evaluator()
        self.evaluator.set_data_int(data[1])
        self.n_vars = len(data[1])

        self.evaluationCount = 0
        # self.penalty_absolute = evaluator.penalty_absolute
    
    def check_phenotype(self, phenotype):
        assert sorted(phenotype)==list(range(self.n_vars)), 'Incorrect phenotype format'
        
    def percentage_true(self):
        # dummy function, don't use
        return 0.5
    
    def total_true(self):
        # dummy function, don't use
        return -1
        
    def evaluate(self, phenotype):
        self.check_phenotype(phenotype)
        self.evaluationCount += 1
        fitness = self.evaluator.penalty_absolute(phenotype)
        # Negative penalty
        return -fitness


    
def print_phenotype(phenotype):
    print("<".join(map(str,phenotype)))
    
def test_evaluator_binary():
    Dint = np.zeros((20,10))
    ip = list(range(10))
    Dint[range(1,11),range(10)]=1
    Dint[range(4),range(4,8)] = 1
    D = (None, Dint, ip)
    evaluator = EvaluatorBinary(D)
    sol = list(range(20))
    assert evaluator.evaluate(sol) == 10, f"Evaluation yielded {evaluator.evaluate(sol)}"
    sol = list(reversed(range(20)))
    assert evaluator.evaluate(sol) == 4, f"Evaluation yielded {evaluator.evaluate(sol)}"
test_evaluator_binary()