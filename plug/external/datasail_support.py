'''
The three helpers datasail_cluster_1d.py imports.

compute_limits and stratification_constraints are DataSAIL's, copied across
unchanged. solve() is NOT theirs, and the difference is the point: theirs takes
a time limit and fourteen threads, so two runs of it can return two different
splits of equal cost. rules.md asks for one shot and exact reproducibility, so
this one runs single-threaded with no limit, and the caller checks that the
status came back optimal rather than merely feasible.
'''

import cvxpy
import numpy as np

def compute_limits(epsilon, total, splits):
    '''DataSAIL's: the lower size bound per split, epsilon under its share'''
    return [int(split * (1 - epsilon) * total) for split in splits]

def stratification_constraints(s_matrix, splits, delta, x):
    '''DataSAIL's: a floor per (split, class). unused here -- nothing is stratified'''
    counts = np.sum(s_matrix, axis=0)
    lower = np.array([[counts[e] * split * (1 - delta) for e in range(s_matrix.shape[1])]
                      for split in splits])
    return (x @ s_matrix) >= lower

def solve(loss, constraints, max_sec, solver, log_file):
    '''
    minimise the loss, to proven optimality, on one thread.

    max_sec and log_file are in the signature because their file passes them;
    both are ignored. a time limit is what makes a solve unreproducible, and the
    objective here is a sum of non-negative terms, so a solution that reaches
    zero is optimal the moment it is found.
    '''
    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    if solver == "SCIP":
        # symmetry detection off. most clusters here carry no similarity at all and
        # differ only by weight, so the symmetry group is enormous and SCIP spent
        # twelve minutes inside bliss without reaching the root LP; with it off,
        # presolve takes 0.3s. it is a search setting, not the model
        problem.solve(solver=cvxpy.SCIP,
                      scip_params={"lp/threads": 1, "parallel/maxnthreads": 1,
                                   "misc/usesymmetry": 0})
    else:
        problem.solve(solver=solver)
    return None if problem.status in ("infeasible", "unbounded") else problem
