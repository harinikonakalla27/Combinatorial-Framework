from typing import Dict, Any, List, Tuple
import numpy as np
from optimization.algorithms.base import (
    GreedyStrategy,
    DynamicProgrammingStrategy,
    BacktrackingStrategy,
    BranchAndBoundStrategy
)

class TSPGreedy(GreedyStrategy):
    """Greedy implementation for Traveling Salesman Problem."""
    
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        distances = problem_instance['distances']
        n = len(distances)
        unvisited = set(range(1, n))
        current = 0  # Start from city 0
        path = [current]
        total_distance = 0
        
        while unvisited:
            next_city = min(unvisited, key=lambda x: distances[current][x])
            total_distance += distances[current][next_city]
            current = next_city
            path.append(current)
            unvisited.remove(current)
        
        # Return to starting city
        total_distance += distances[current][0]
        path.append(0)
        
        return {
            'path': path,
            'distance': total_distance,
            'strategy': 'greedy'
        }
    
    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        path = solution['path']
        distances = problem_instance['distances']
        
        # Check if path starts and ends at city 0
        if path[0] != 0 or path[-1] != 0:
            return False
            
        # Check if all cities are visited exactly once
        visited = set(path[:-1])  # Exclude the last city (return to start)
        if len(visited) != len(distances):
            return False
            
        return True

class TSPDynamic(DynamicProgrammingStrategy):
    """Dynamic Programming implementation for TSP."""
    
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        distances = problem_instance['distances']
        n = len(distances)
        all_points = (1 << n) - 1
        
        # Initialize memoization table
        memo = {}
        
        def dp(mask: int, pos: int) -> Tuple[float, List[int]]:
            if mask == all_points and pos == 0:
                return 0, [0]
            
            if (mask, pos) in memo:
                return memo[(mask, pos)]
            
            ans = float('inf')
            best_path = []
            
            for city in range(n):
                if (mask & (1 << city)) == 0:  # if city is not visited
                    new_mask = mask | (1 << city)
                    dist, path = dp(new_mask, city)
                    total_dist = distances[pos][city] + dist
                    if total_dist < ans:
                        ans = total_dist
                        best_path = [pos] + path
            
            memo[(mask, pos)] = ans, best_path
            return ans, best_path
        
        total_distance, path = dp(1, 0)  # Start with only city 0 visited
        
        return {
            'path': path,
            'distance': total_distance,
            'strategy': 'dynamic'
        }
    
    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        # Use the same validation as greedy
        return TSPGreedy().validate_solution(solution, problem_instance)

# You can implement Backtracking and Branch & Bound strategies similarly