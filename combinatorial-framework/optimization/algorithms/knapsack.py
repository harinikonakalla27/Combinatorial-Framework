from typing import Dict, Any, List
import numpy as np
from optimization.algorithms.base import (
    GreedyStrategy,
    DynamicProgrammingStrategy,
    BacktrackingStrategy,
    BranchAndBoundStrategy
)

class KnapsackGreedy(GreedyStrategy):
    """Greedy implementation for 0/1 Knapsack Problem."""
    
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        weights = problem_instance['weights']
        values = problem_instance['values']
        capacity = problem_instance['capacity']
        
        # Calculate value/weight ratio for each item
        ratios = [(i, values[i]/weights[i]) for i in range(len(weights))]
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        total_value = 0
        total_weight = 0
        selected_items = []
        
        for item, _ in ratios:
            if total_weight + weights[item] <= capacity:
                selected_items.append(item)
                total_value += values[item]
                total_weight += weights[item]
        
        return {
            'selected_items': selected_items,
            'total_value': total_value,
            'total_weight': total_weight,
            'strategy': 'greedy'
        }
    
    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        weights = problem_instance['weights']
        capacity = problem_instance['capacity']
        selected_items = solution['selected_items']
        
        # Check if total weight is within capacity
        total_weight = sum(weights[i] for i in selected_items)
        if total_weight > capacity:
            return False
            
        # Check if items are valid indices
        if any(i >= len(weights) for i in selected_items):
            return False
            
        return True

class KnapsackDynamic(DynamicProgrammingStrategy):
    """Dynamic Programming implementation for 0/1 Knapsack Problem."""
    
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        weights = problem_instance['weights']
        values = problem_instance['values']
        capacity = problem_instance['capacity']
        n = len(weights)
        
        # Initialize DP table
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        keep = [[False for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    if values[i-1] + dp[i-1][w-weights[i-1]] > dp[i-1][w]:
                        dp[i][w] = values[i-1] + dp[i-1][w-weights[i-1]]
                        keep[i][w] = True
                    else:
                        dp[i][w] = dp[i-1][w]
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Reconstruct solution
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if keep[i][w]:
                selected_items.append(i-1)
                w -= weights[i-1]
        
        return {
            'selected_items': selected_items[::-1],
            'total_value': dp[n][capacity],
            'total_weight': sum(weights[i] for i in selected_items),
            'strategy': 'dynamic'
        }
    
    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        # Use the same validation as greedy
        return KnapsackGreedy().validate_solution(solution, problem_instance)

# You can implement Backtracking and Branch & Bound strategies similarly