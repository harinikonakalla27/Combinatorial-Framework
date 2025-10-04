import pytest
import numpy as np
from optimization.algorithms.tsp import TSPGreedy, TSPDynamic
from optimization.algorithms.knapsack import KnapsackGreedy, KnapsackDynamic

def test_tsp_greedy():
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    problem_instance = {'distances': distances}
    solver = TSPGreedy()
    solution = solver.solve(problem_instance)
    
    assert solver.validate_solution(solution, problem_instance)
    assert len(solution['path']) == len(distances) + 1  # All cities plus return to start
    assert solution['path'][0] == solution['path'][-1] == 0  # Start and end at city 0

def test_knapsack_dynamic():
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 10
    problem_instance = {
        'weights': weights,
        'values': values,
        'capacity': capacity
    }
    solver = KnapsackDynamic()
    solution = solver.solve(problem_instance)
    
    assert solver.validate_solution(solution, problem_instance)
    assert solution['total_weight'] <= capacity
    assert all(i < len(weights) for i in solution['selected_items'])