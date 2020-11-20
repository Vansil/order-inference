import os
import numpy as np
import argparse

import order.algorithm

N_vars_total = 1479

ALGORITHMS = [
        'weighted_trueskill',
        'sun_ensemble',
        'sun_greedy',
        'sort_trueskill',
        'sort_socialagony',
        'sort_pagerank',
        'edmond',
        'edmond_sparse',
        'evolutionstrategy_binary',
        'evolutionstrategy_penalty',
        'random'
    ]
ALGS_REVERSABLE = [
        'sort_trueskill',
        'sort_socialagony',
        'sort_pagerank',
        'edmond',
        'edmond_sparse',
    ]

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='../Output/02order_wts/',
    help="Base directory for experiment output")
parser.add_argument('-N', '--nvars', type=int, default=N_vars_total,
    help="Number of variables in subset")
parser.add_argument('-M', '--rep', type=int, default=5,
    help="Number of repetitions of the same experiment")
parser.add_argument('-R', '--reversed', type=int, default=0,
    help="If the graph should be reversed for score-ordering methods")
parser.add_argument('--algorithm', type=str, default='run',
    choices=ALGORITHMS,
    help='Algorithm to find the order')


def main(args):
    # Determine files and directories
    dir_base = args.dir
    file_csv = os.path.join(dir_base, 'results.csv')
    print(f"Results file at {file_csv}")
    file_data = '../data/kemmeren/orig.p'
    dir_output = os.path.join(dir_base, args.algorithm, str(args.nvars), str(args.rep), str(args.reversed), 'output')
    # Skip if dir exists
    if os.path.exists(dir_output):
        print(f"Skipping because output directory already exists: {dir_output}")
        return
    # Create files and directories
    os.makedirs(dir_output, exist_ok=True)
    if not os.path.exists(file_csv):
        print(f"Creating results file at {file_csv}")
        with open(file_csv, 'w') as f:
            f.write('N,M,algorithm,penalty,penaltyratio,timereal,timeuser,timesystem')
    
    # Determine intervention ids
    np.random.seed(42+args.nvars*69+args.rep) # select same subset for all experiments;
    int_ids = list(np.random.choice(range(N_vars_total), args.nvars, replace=False))

    # Apply method
    if args.algorithm == 'weighted_trueskill':
        O = order.algorithm.update_weighted_trueskill(
            file_data=file_data,
            dir_output=dir_output,
            int_ids=int_ids)
    if args.algorithm == 'sun_ensemble':
        O = order.algorithm.sun(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids, 
            model_type='ensembling')
    if args.algorithm == 'sun_greedy':
        O = order.algorithm.sun(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids, 
            model_type='trueskill-greedy')
    if args.algorithm == 'sort_trueskill':
        O = order.algorithm.sort_score(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids, 
            score_type='trueskill',
            reverse=args.reversed==1)
    if args.algorithm == 'sort_socialagony':
        O = order.algorithm.sort_score(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids, 
            score_type='socialagony',
            reverse=args.reversed==1)
    if args.algorithm == 'sort_pagerank':
        O = order.algorithm.sort_score(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids, 
            score_type='pagerank',
            reverse=args.reversed==1)
    if args.algorithm == 'edmond':
        O = order.algorithm.edmond(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids,
            reverse=args.reversed==1)
    if args.algorithm == 'edmond_sparse':
        O = order.algorithm.edmond_sparse(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids,
            reverse=args.reversed==1)
    if args.algorithm == 'evolutionstrategy_binary':
        O = order.algorithm.evolution_strategy(
            file_data=file_data, 
            dir_output=dir_output, 
            int_ids=int_ids,
            fitness_type='binary')
    if args.algorithm == 'evolutionstrategy_penalty':
        O = order.algorithm.evolution_strategy(
            file_data=file_data, 
            dir_output=os.path.join(dir_output, 'evolutionstrategy_penalty/'), 
            int_ids=int_ids, 
            fitness_type='continuous')
    if args.algorithm == 'random':
        O = order.algorithm.random(int_ids)

    # Evaluate method
    from order.evaluate import Evaluator
    evaluator = Evaluator(file_data)
    penalty, penalty_ratio = evaluator.penalty_absolute(O, int_ids=int_ids, return_ratio=True)

    print("ORDER",O,"penalty",penalty, "ratio", penalty_ratio)

    # Write results to csv
    with open(file_csv, 'a') as f:
        if args.reversed==1:
            f.write(f"\n{args.nvars},{args.rep},{args.algorithm}_reversed,{penalty},{penalty_ratio},")
        else:
            f.write(f"\n{args.nvars},{args.rep},{args.algorithm},{penalty},{penalty_ratio},")


def generate_bash(file_out='run_exp_order.sh'):
    exp = "echo '======== NEW EXPERIMENT N{1}R{3}M{2} - {0} ============='\n(time -p python exp_order.py --dir ../Output/04order/ --algorithm {0} -N {1} -M {2} -R {3}) 2> tmp_time\npython read_runtime.py --csvfile ../Output/04order/results.csv\n\n"

    with open(file_out, 'w') as f:
        for N in [2,5,10,20,30,50,75]: #[2,5,10,20,30,50,75, 100, 125, 150, 175, 200, 300, 500, 750, 1000, 1479]:
            for algorithm in [
                    'sun_ensemble',
                    'sun_greedy',
                    'sort_trueskill',
                    'sort_socialagony',
                    'sort_pagerank',
                    'edmond',
                    'edmond_sparse',
                    'evolutionstrategy_binary',
                    'evolutionstrategy_penalty',
                    'random']:
                reverse = [0,1] if algorithm in ALGS_REVERSABLE else [0]
                for R in reverse:
                    for M in range(5):
                        f.write(exp.format(algorithm, N, M, R))

def generate_bash_manual(file_out='run_exp_orderB.sh'):
    exp = "echo '======== NEW EXPERIMENT N{1}R{3}M{2} - {0} ============='\n(time -p python exp_order.py --dir ../Output/04order/ --algorithm {0} -N {1} -M {2} -R {3}) 2> tmp_time\npython read_runtime.py --csvfile ../Output/04order/results.csv\n\n"

    with open(file_out, 'w') as f:
        for N in [100, 125, 150, 175, 200, 300, 500, 750, 1000, 1479]: #[2,5,10,20,30,50,75, 100, 125, 150, 175, 200, 300, 500, 750, 1000, 1479]:
            for algorithm in [
                    'sun_ensemble',
                    'sun_greedy',
                    'sort_trueskill',
                    'sort_socialagony',
                    'sort_pagerank',
                    'edmond',
                    'edmond_sparse',
                    'evolutionstrategy_binary',
                    'evolutionstrategy_penalty',
                    'random']:
                if algorithm in ['edmond', 'sun_ensemble', 'evolutionstrategy_binary'] and N>125:
                    continue
                if algorithm in ['sun_greedy'] and N>200:
                    continue
                reverse = [0,1] if algorithm in ALGS_REVERSABLE else [0]
                for R in reverse:
                    for M in range(5):
                        f.write(exp.format(algorithm, N, M, R))

def generate_bash_wt(file_out='run_exp_order_wt.sh'):
    exp = "echo '======== NEW EXPERIMENT N{1} - {0} ============='\n(time -p python exp_order.py --algorithm {0} -N {1}) 2> tmp_time\npython read_runtime.py\n\n"

    with open(file_out, 'w') as f:
        for N in [2,5,10,20,30,50,75, 100, 125, 150, 175, 200, 300, 500, 750, 1000, 1479]:
            algorithm = 'weighted_trueskill'
            f.write(exp.format(algorithm, N))
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
