"""
Experiment to find order in synthetic data using an Evolution Strategy
"""

import numpy as np
import random
import pickle
import argparse
import os
from copy import copy

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True,
    help="Base directory for experiment output")
parser.add_argument('--mode', type=str, default='run',
    choices=['run', 'evaluate'],
    help='Mode of operation')
parser.add_argument('--variation', type=str, default='direct',
    choices=['direct','cyclic', 'cyclicmutate'],
    help="Type of recombination to use")
parser.add_argument('--nvars', type=int, default=10,
    help="Number of variables in synthetic data")
parser.add_argument('--nruns', type=int, default=5,
    help="Number of experiments per setting")

def main(args):
    # Determine directories
    dir_base = args.dir
    dir_output = os.path.join(dir_base, 'output')
    dir_results = os.path.join(dir_base, 'results')

    if args.mode == 'run':
        # Create directories
        for d in [dir_output, dir_results]: 
            os.makedirs(d, exist_ok=True)
        print("Directories created")
        # Run experiment
        for run_index in range(args.nruns):
            file_out = os.path.join(dir_output, f"vars{args.nvars}_{args.variation}_index{run_index}.csv")
            print("Experiment: ",file_out)
            # generate data
            data = generate_perfect_gt(args.nvars, 0.05)
            while data[1].sum()==0:
                data = generate_perfect_gt(args.nvars, 0.05)
            evaluator = EvaluatorBinary(data)
            perfect_fitness = data[1].sum()
            # set up solver
            solver = Solver(evaluator)
            if 'cyclic' in args.variation:
                solver.recomb_mode = 'cyclic'
            else:
                solver.recomb_mode = 'direct'
            if 'mutate' in args.variation:
                solver.mutation_swap_rate = 0.05
            results = solver.run(verbose=True) # [(generation, best fitness, diversity, best genotype)]
            # Write to file
            with open(file_out,'w') as f:
                for gen, fit, div, geno in results:
                    fit_perc = fit / perfect_fitness
                    div_perc = div / args.nvars
                    geno_str = ";".join(map(str,geno))
                    f.write(f"{gen},{fit_perc},{div_perc},{geno_str}\n")
            print(f"Done")
            
    elif args.mode == 'evaluate':
        # TODO
        pass

def generate_perfect_gt(n_vars, p):
    """
    Generate square gt matrix for which there is a perfect order
    Arguments:
        n_vars: dimension of matrix
        p: probability of true relation
    """
    gt_matrix = np.tril(
        np.random.choice([True,False],n_vars**2,p=[p,1-p]).reshape((n_vars,n_vars)),
        -1
    )
    return (None, gt_matrix, list(range(n_vars)))

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
    
def print_phenotype(phenotype):
    print("<".join(map(str,phenotype)))

class Solver(object):
    '''
    classdocs
    '''
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.n_vars = evaluator.n_vars

        self.population_size = 100
        self.offspring_size = 400
        self.elite_size = 5
        
        self.recomb_rate = 1
        self.recomb_mode = 'cyclic'
        
        self.inversion_size = self.mutation_size_per_relation_swap()
        self.mutation_inversion_rate = 0#.05 / self.inversion_size # make approx. 1 out of 5 relation swaps
        self.mutation_swap_rate = 0.05
        
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
                print("{:03d}\t\t{:d}\t\t{:.2f}".format(
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)