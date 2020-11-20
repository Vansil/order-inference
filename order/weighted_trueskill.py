import numpy as np
from copy import deepcopy
import random
import pickle
import networkx as nx
from collections import defaultdict

import evaluation
from trueskill import Rating, rate_1vs1
from order.evaluate import Evaluator


def get_players_score(players,n_sigma):
	relative_score = {}
	for k in players.keys():
		relative_score[k] = players[k].mu - n_sigma * players[k].sigma
	return relative_score

def update_rating(x, y, f):
    x_, y_ = rate_1vs1(x, y)
    x = Rating(
        x.mu + f * (x_.mu - x.mu),
        x.sigma * np.power(x_.sigma / x.sigma, f)
    )
    y = Rating(
        y.mu + f * (y_.mu - y.mu),
        y.sigma * np.power(y_.sigma / y.sigma, f)
    )
    return x, y

def edge_triples(edges):
    for u,v,weight_dict in edges:
        yield u, v, weight_dict['weight']

def new_players(pairs):
    players = {}
    for u,v,_ in pairs:
        if u not in players:
            players[u] = Rating()
        if v not in players:
            players[v] = Rating()
    return players        

def compute_weighted_trueskill(pairs, players_, factor=1):
    """
    pairs: [(effect, cause, {'weight': w})]
    factor: multiply weight with factor
    """
    players = deepcopy(players_)
    if not players:
        players = new_players(pairs)
    # update in random order
    random.shuffle(pairs)
    for u, v, weight in edge_triples(pairs):
        players[v],players[u] = update_rating(players[v],players[u], f=abs(weight)*factor)
    
    return players

def compute_unweighted_trueskill(pairs, players_):
    """
    pairs: [(effect, cause, {'weight': w})]
    """
    players = deepcopy(players_)
    if not players:
        players = new_players(pairs)
    # update in random order
    random.shuffle(pairs)
    for u, v, weight in edge_triples(pairs):
        players[v],players[u] = update_rating(players[v],players[u], f=1)
    
    return players

def run(file_data, int_ids):
    """
    Define order by sorting the iterated update-weighted TrueSkill scores
    Parameters:
    """
    evaluator = Evaluator(file_data)
    penalty_absolute = evaluator.penalty_absolute

    # Algorithm parameters
    WEIGHTED = True
    WEIGHT_FACTOR = 5
    GT_THRESHOLD = .5
    # Early stopping parameters
    MAX_ITER = 30
    MAX_NO_IMPROVE = 10 # stop after 10 iterations without improved penalty

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
    A_weighted = deepcopy(data_int)
    if GT_THRESHOLD != 0:
        A_weighted[np.logical_not(A)] = 0
    g = nx.from_numpy_matrix(A_weighted, create_using=nx.DiGraph)
    nx.relabel_nodes(g, {i: int_ids[i] for i in range(len(int_ids))}, copy=False)
    pairs = list(g.edges(data=True))
    players = new_players(pairs)

    # Track best order
    best_order = None
    min_penalty = 100
    no_improve_count = 0
    # Iterate
    for _ in range(MAX_ITER):
        # Perform update step
        if WEIGHTED:
            players = compute_weighted_trueskill(pairs, players, factor=WEIGHT_FACTOR)
        else:
            players = compute_unweighted_trueskill(pairs, players)
        # Evaluate
        relative_scores = get_players_score(players,n_sigma = 3)
        scores = defaultdict(
            lambda:0,
            relative_scores)
        order = np.argsort([scores[i] for i in int_ids])[::-1]
        order = list(np.array(int_ids)[order])
        penalty = penalty_absolute(order, int_ids=int_ids)
        # Track best order
        if penalty < min_penalty:
            no_improve_count = 0
            penalty = min_penalty
            best_order = deepcopy(order)
        else:
            no_improve_count += 1
            # early stopping criterium
            if no_improve_count > MAX_NO_IMPROVE: 
                break
    
    return best_order