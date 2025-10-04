from typing import Dict, Any, List
import numpy as np
from optimization.algorithms.base import (
    GreedyStrategy,
    DynamicProgrammingStrategy,
    BacktrackingStrategy
)

class KnapsackGreedy(GreedyStrategy):
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        weights = problem_instance['weights']
        values = problem_instance['values']
        capacity = problem_instance['capacity']
        
        # Calculate value-to-weight ratios
        ratios = [(v/w, i) for i, (v, w) in enumerate(zip(values, weights))]
        ratios.sort(reverse=True)  # Sort by ratio in descending order
        
        selected_items = []
        total_weight = 0
        total_value = 0
        
        for ratio, idx in ratios:
            if total_weight + weights[idx] <= capacity:
                selected_items.append(idx)
                total_weight += weights[idx]
                total_value += values[idx]
        
        return {
            'selected_items': selected_items,
            'total_weight': float(total_weight),
            'total_value': float(total_value),
            'strategy': 'greedy'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        weights = problem_instance['weights']
        capacity = problem_instance['capacity']
        selected_items = solution['selected_items']
        
        total_weight = sum(weights[i] for i in selected_items)
        return total_weight <= capacity

class KnapsackDynamic(DynamicProgrammingStrategy):
    def __init__(self):
        super().__init__()
        self.memo = {}

    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        weights = problem_instance['weights']
        values = problem_instance['values']
        capacity = problem_instance['capacity']
        n = len(weights)
        
        # Create DP table
        dp = np.zeros((n + 1, int(capacity) + 1))
        
        # Build table bottom-up
        for i in range(1, n + 1):
            for w in range(int(capacity) + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        values[i-1] + dp[i-1][int(w-weights[i-1])],
                        dp[i-1][w]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtrack to find selected items
        selected_items = []
        w = int(capacity)
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_items.append(i-1)
                w = int(w - weights[i-1])
        
        selected_items.reverse()
        total_weight = sum(weights[i] for i in selected_items)
        total_value = sum(values[i] for i in selected_items)
        
        return {
            'selected_items': selected_items,
            'total_weight': float(total_weight),
            'total_value': float(total_value),
            'strategy': 'dynamic'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        return KnapsackGreedy().validate_solution(solution, problem_instance)

class KnapsackBacktracking(BacktrackingStrategy):
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        weights = problem_instance['weights']
        values = problem_instance['values']
        capacity = problem_instance['capacity']
        n = len(weights)
        
        best_value = 0
        best_solution = []
        
        def bound(items: List[int], curr_value: float, curr_weight: float, idx: int) -> float:
            """Calculate upper bound for remaining capacity using fractional knapsack"""
            if curr_weight >= capacity:
                return curr_value
            
            bound_value = curr_value
            bound_weight = curr_weight
            
            # Sort remaining items by value/weight ratio
            remaining = [(values[i]/weights[i], values[i], weights[i]) 
                        for i in range(idx, n) if i not in items]
            remaining.sort(reverse=True)
            
            for ratio, v, w in remaining:
                if bound_weight + w <= capacity:
                    bound_value += v
                    bound_weight += w
                else:
                    # Take fraction of last item
                    remaining_capacity = capacity - bound_weight
                    bound_value += v * (remaining_capacity / w)
                    break
            
            return bound_value
        
        def backtrack(items: List[int], curr_value: float, curr_weight: float, idx: int) -> None:
            nonlocal best_value, best_solution
            
            if curr_weight > capacity:
                return
                
            if curr_value > best_value:
                best_value = curr_value
                best_solution = items.copy()
                
            if idx == n:
                return
                
            # Calculate bound
            if bound(items, curr_value, curr_weight, idx) <= best_value:
                return  # Prune this branch
            
            # Include item at idx
            items.append(idx)
            backtrack(
                items, 
                curr_value + values[idx],
                curr_weight + weights[idx],
                idx + 1
            )
            items.pop()
            
            # Exclude item at idx
            backtrack(items, curr_value, curr_weight, idx + 1)
        
        # Start backtracking
        backtrack([], 0, 0, 0)
        
        total_weight = sum(weights[i] for i in best_solution)
        total_value = sum(values[i] for i in best_solution)
        
        return {
            'selected_items': best_solution,
            'total_weight': float(total_weight),
            'total_value': float(total_value),
            'strategy': 'backtrack'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        return KnapsackGreedy().validate_solution(solution, problem_instance)