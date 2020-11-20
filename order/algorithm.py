"""
Algorithms to infer topological order from interventional data
"""

import pickle
import numpy as np
import os
import sys
import networkx as nx
from collections import defaultdict
from copy import deepcopy

import evaluation.ground_truth
from order import es
from order.evaluate import Evaluator
from order import weighted_trueskill

# Sun algorithm mess
sys.path.append('order/sun/')
import remove_cycle_edges_by_hierarchy as sunalg
from file_io import write_dict_to_file
from true_skill import graphbased_trueskill
from compute_social_agony import compute_social_agony



def update_weighted_trueskill(file_data='../data/kemmeren/orig.p', dir_output='../Output/02ordertest/updateweightedtrueskill/', int_ids=None):
    """
    Define order by sorting the iterated update-weighted TrueSkill scores
    Parameters:
    """

    # Define files
    file_order = os.path.join(dir_output, 'order.csv')
    os.makedirs(dir_output, exist_ok=True)

    # Run algorithm
    order = weighted_trueskill.run(file_data, int_ids)
    
    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))
    return order


def sun(file_data='../data/kemmeren/orig.p', dir_output='../Output/00order_size/sun/', int_ids=None, model_type='ensembling'):
    """
    Aply Sun's algorithm
    NOTE: if there are multiple weakly connected components in the infered binary ground-truth, 
        this algorithm puts them in arbitrary respective order

    Returns:
        order of integer ids
    """
    os.makedirs(dir_output, exist_ok=True)

    GT_THRESHOLD = 0.8 # percent
    print(f"Set ground-truth threshold at {GT_THRESHOLD}")
    file_edgelist = os.path.join(dir_output, 'graph.edges')
    if model_type=='ensembling':
        file_reduced_edgelist = os.path.join(dir_output, 'graph_removed_by_H-Voting.edges')
    elif model_type == 'trueskill-greedy':
        file_reduced_edgelist = os.path.join(dir_output, 'graph_removed_by_TS_G.edges')
    else:
        raise NotImplementedError
    file_order = os.path.join(dir_output, 'order.csv')

    # select only mutant-mutant data
    data = pickle.load(open(file_data, 'rb'))
    intpos = data[2]
    data_int = data[1][intpos]
    del data
    # select subset of variables
    data_int = data_int[int_ids][:, int_ids]

    # Create graph based on absolute threshold binary ground-truth
    # NOTE: Graph with edges from effect to cause, such that high score relates to early in order
    _, A = evaluation.ground_truth.abs_percentile(data_int, range(len(data_int)), percentile=GT_THRESHOLD, dim='full')
    G_gt = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G_gt, {i: int_ids[i] for i in range(len(int_ids))}, copy=False)
    # Write edge list to file
    nx.write_edgelist(G_gt, file_edgelist, data=False)

    # Run Sun
    if model_type == 'ensembling':
        sunalg.breaking_cycles_by_hierarchy_performance(
            graph_file=file_edgelist,
            gt_file=None,
            players_score_name=model_type)
    elif model_type == 'trueskill-greedy':
        players_score_dict  = sunalg.computing_hierarchy(
            graph_file=file_edgelist,
            players_score_func_name='trueskill',
            nodetype = int)
        g = nx.read_edgelist(file_edgelist,create_using = nx.DiGraph(),nodetype = int)
        e1 = sunalg.scc_based_to_remove_cycle_edges_iterately(g, players_score_dict)
        sunalg.write_pairs_to_file(e1, file_reduced_edgelist)
    
    # Remove edges from adjacency matrix
    reduced_edgelist = nx.read_edgelist(file_reduced_edgelist, nodetype=int, create_using=nx.DiGraph).edges
    for edge in reduced_edgelist: 
        A[int_ids.index(edge[0]),int_ids.index(edge[1])]=False
    # Infer topological order
    G_reduced = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.relabel_nodes(G_reduced, {i: int_ids[i] for i in range(len(int_ids))}, copy=False)
    order = list(reversed(list(nx.topological_sort(G_reduced))))
    
    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))
    return order


def sort_score(file_data='../data/kemmeren/orig.p', dir_output='../Output/00order_size/sortscore/', int_ids=None, score_type='trueskill', reverse=False, return_scores=False):
    """
    Define order by sorting some hierarchy score
    Parameters:
        score_type: trueskill, socialagony, pagerank
        reverse=Forward=E->C: False is default, edges go Cause->Effect (Backward interpretation) and a high rank (likely cause) corresponds to:
            TS: high score (causes are the winner)
            SA: high score (causes are higher in the hierarchy)
            PR: high score (causes are often refered to by effects [if there is a strong fan-out effect, we might reverse this])
    """
    os.makedirs(dir_output, exist_ok=True)

    GT_THRESHOLD=0.8 # percent
    print(f"Set ground-truth threshold at {GT_THRESHOLD}")
    file_edgelist = os.path.join(dir_output, 'graph.edges')
    file_order = os.path.join(dir_output, 'order.csv')

    # select only mutant-mutant data
    data = pickle.load(open(file_data, 'rb'))
    intpos = data[2]
    data_int = data[1][intpos]
    del data
    # select subset of variables
    data_int = data_int[int_ids][:, int_ids]

    # Create graph based on absolute threshold binary ground-truth
    # NOTE: Graph with edges from effect to cause, such that high score relates to early in order
    _, A = evaluation.ground_truth.abs_percentile(data_int, range(len(data_int)), percentile=GT_THRESHOLD, dim='full')
    # Transpose adjacency matrix if edges are reversed
    if reverse:
        A = A.T
    g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    # TODO: IS DIT ECHT CORRECT???
    # (edit 27 mei; was copy=False en zonder 'g = ')
    g = nx.relabel_nodes(g, {i: int_ids[i] for i in range(len(int_ids))}, copy=True)
    # Write edge list to file
    nx.write_edgelist(g, file_edgelist, data=False)

    # Compute scores
    if score_type == "pagerank":
        print("computing pagerank...")
        scores = defaultdict(
            lambda:0,
            nx.pagerank(g, alpha = 0.85))
    elif score_type == 'socialagony':
        agony_file = os.path.join(dir_output,"graph_socialagony.txt")
        print("start computing socialagony...")
        scores = defaultdict(
            lambda:0,
            compute_social_agony(
                graph_file=file_edgelist, 
                agony_path = "order/sun/agony/agony "))
        print("write socialagony to file: %s" % agony_file)
    elif score_type == 'trueskill':
        trueskill_file = os.path.join(dir_output,"graph_trueskill.txt")
        print("start computing trueskill...")
        scores = defaultdict(
            lambda:0,
            graphbased_trueskill(g))
        print("write trueskill to file: %s" % trueskill_file)
        write_dict_to_file(scores, trueskill_file)
    else:
        raise NotImplementedError

    # Determine order
    order = np.argsort([scores[i] for i in int_ids])[::-1]
    order = list(np.array(int_ids)[order])
    # Reverse order if edges are reversed
    if reverse:
        order = order[::-1]

    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))

    if return_scores:
        return order, scores
    else:
        return order


def edmond(file_data='../data/kemmeren/orig.p', dir_output='../Output/00order_size/edmond/', int_ids=None, reverse=False):
    """
    Define order by Edmond's algorithm for the optimal branching problem
    NOTE: prohibitively expensive for fully-connected graphs of over 300 nodes
    reversed: if False, edges go Effect->Cause
    """
    os.makedirs(dir_output, exist_ok=True)

    file_edgelist = os.path.join(dir_output, 'graph.edges')
    file_order = os.path.join(dir_output, 'order.csv')

    # select only mutant-mutant data
    data = pickle.load(open(file_data, 'rb'))
    intpos = data[2]
    data_int = data[1][intpos]
    del data
    # select subset of variables
    data_int = data_int[int_ids][:, int_ids]
    # Transpose adjacency matrix if edges are reversed
    if reverse:
        data_int = data_int.T

    # Create graph and solve arborescence problem
    G = nx.from_numpy_matrix(abs(data_int), create_using=nx.DiGraph())
    nx.relabel_nodes(G, {i: int_ids[i] for i in range(len(int_ids))}, copy=False)
    ed = nx.algorithms.tree.branchings.Edmonds(G)
    G_arb = ed.find_optimum()

    # Infer topological order
    order = list(reversed(list(nx.topological_sort(G_arb))))
    # Reverse order if edges are reversed
    if reverse:
        order = order[::-1]
    
    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))
    return order


def edmond_sparse(file_data='../data/kemmeren/orig.p', dir_output='../Output/00order_size/edmond/', int_ids=None, edges_per_node=10, reverse=False):
    """
    Define order by Edmond's algorithm for the optimal branching problem
    edges_per_node: select on average this number of edges per node
    reversed: if False, edges go Effect->Cause
    """
    os.makedirs(dir_output, exist_ok=True)

    file_edgelist = os.path.join(dir_output, 'graph.edges')
    file_order = os.path.join(dir_output, 'order.csv')

    # select only mutant-mutant data
    data = pickle.load(open(file_data, 'rb'))
    intpos = data[2]
    data_int = data[1][intpos]
    del data
    # select subset of variables
    data_int = data_int[int_ids][:, int_ids]
    # make sparse
    N = len(data_int)
    edges_per_node = min(edges_per_node, N)
    perc = 1-edges_per_node/N
    T = np.percentile(abs(data_int).flatten(), perc*100)
    data_sparse = abs(data_int)
    data_sparse[abs(data_int)<T] = 0
    # Transpose adjacency matrix if edges are reversed
    if reverse:
        data_sparse = data_sparse.T

    # Create graph and solve arborescence problem
    G = nx.from_numpy_matrix(data_sparse, create_using=nx.DiGraph())
    nx.relabel_nodes(G, {i: int_ids[i] for i in range(len(int_ids))}, copy=False)
    ed = nx.algorithms.tree.branchings.Edmonds(G)
    G_arb = ed.find_optimum()

    # Infer topological order
    order = list(reversed(list(nx.topological_sort(G_arb))))
    # Reverse order if edges are reversed
    if reverse:
        order = order[::-1]
    
    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))
    return order


def evolution_strategy(file_data='../data/kemmeren/orig.p', dir_output='../Output/00order_size/evolutionstrategy/', int_ids=None, fitness_type='binary'):
    """
    Parameters:
        fitness_type: ['binary', 'continuous']
    """
    if fitness_type == 'binary':
        GT_THRESHOLD = 1 # absolute
        print(f"Set ground-truth threshold at {GT_THRESHOLD} absolute")

    os.makedirs(dir_output, exist_ok=True)
    file_order = os.path.join(dir_output, 'order.csv')
    file_output = os.path.join(dir_output, 'output.p')

    # select only mutant-mutant data
    data = pickle.load(open(file_data, 'rb'))
    intpos = data[2]
    data_int = data[1][intpos]
    del data
    # select subset of variables
    data_int = data_int[int_ids][:, int_ids]

    # Set data type for ES (NOTE: interv. pos is mapped to range)
    D = (None, data_int, list(range(len(int_ids))))
    if fitness_type == 'binary':
        evaluator = es.EvaluatorBinary(D, threshold=GT_THRESHOLD)
    elif fitness_type == 'continuous':
        evaluator = es.EvaluatorContinuous(D)
    solver = es.Solver(evaluator, nvars=len(int_ids))
    results = solver.run(verbose=True)

    # Store results
    pickle.dump(results, open(file_output, 'wb'))
    
    # Map order to intervention ids
    order = results[-1][3]
    order = list(np.array(int_ids)[order])

    # Output order
    with open(file_order, 'w') as f:
        f.write(",".join(str(i) for i in order))
    return order


def random(int_ids):
    return list(np.random.permutation(int_ids))
