from typing import Dict, Any, List, Tuple
import numpy as np
from optimization.algorithms.base import (
    GreedyStrategy,
    DynamicProgrammingStrategy,
    BacktrackingStrategy
)

class TSPGreedy(GreedyStrategy):
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        distances = problem_instance['distances']
        n = len(distances)
        unvisited = set(range(1, n))
        current = 0
        path = [current]
        total_distance = 0
        
        while unvisited:
            next_city = min(unvisited, key=lambda x: distances[current][x])
            total_distance += distances[current][next_city]
            current = next_city
            path.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_distance += distances[current][0]
        path.append(0)
        
        return {
            'path': path,
            'distance': float(total_distance),
            'strategy': 'greedy'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        # Basic structure checks
        if not isinstance(solution, dict) or 'path' not in solution:
            return False

        path = solution.get('path')
        distances = problem_instance.get('distances') if isinstance(problem_instance, dict) else None

        # Validate path content
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            # Must at least start and end at the depot 0
            return False

        # Validate start/end are at depot 0
        if path[0] != 0 or path[-1] != 0:
            return False

        # Validate distances matrix
        if distances is None:
            return False

        num_cities = len(distances)
        # All cities 0..n-1 should be visited exactly once (excluding final return)
        visited = list(path[:-1])
        try:
            visited_set = set(int(c) for c in visited)
        except Exception:
            return False

        if len(visited) != num_cities:
            return False

        if visited_set != set(range(num_cities)):
            return False

        return True

class TSPDynamic(DynamicProgrammingStrategy):
    def __init__(self):
        self.memo = {}

    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        distances = problem_instance['distances']
        n = len(distances)
        all_points = (1 << n) - 1
        
        def dp(mask: int, pos: int) -> tuple[float, List[int]]:
            if mask == all_points and pos == 0:
                return 0, [0]
            
            if (mask, pos) in self.memo:
                return self.memo[(mask, pos)]
            
            ans = float('inf')
            best_path = []
            
            for city in range(n):
                if (mask & (1 << city)) == 0:
                    new_mask = mask | (1 << city)
                    dist, path = dp(new_mask, city)
                    total_dist = distances[pos][city] + dist
                    if total_dist < ans:
                        ans = total_dist
                        best_path = [pos] + path
            
            self.memo[(mask, pos)] = ans, best_path
            return ans, best_path

        total_distance, path = dp(1, 0)
        return {
            'path': path,
            'distance': float(total_distance),
            'strategy': 'dynamic'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        return TSPGreedy().validate_solution(solution, problem_instance)

class TSPBacktracking(BacktrackingStrategy):
    def solve(self, problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        distances = problem_instance['distances']
        n = len(distances)
        visited = [False] * n
        path = [0]  # Start from city 0
        visited[0] = True
        best_path = None
        best_distance = float('inf')
        
        def backtrack(curr_path: List[int], curr_dist: float) -> None:
            nonlocal best_path, best_distance
            
            if len(curr_path) == n:
                # Return to start
                total_dist = curr_dist + distances[curr_path[-1]][0]
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_path = curr_path + [0]
                return
            
            curr_city = curr_path[-1]
            for next_city in range(n):
                if not visited[next_city]:
                    new_dist = curr_dist + distances[curr_city][next_city]
                    if new_dist < best_distance:  # Pruning
                        visited[next_city] = True
                        backtrack(curr_path + [next_city], new_dist)
                        visited[next_city] = False
        
        backtrack([0], 0)
        
        if best_path is None:
            best_path = list(range(n)) + [0]
            best_distance = sum(distances[best_path[i]][best_path[i+1]] for i in range(n))
        
        return {
            'path': best_path,
            'distance': float(best_distance),
            'strategy': 'backtrack'
        }

    def validate_solution(self, solution: Dict[str, Any], problem_instance: Dict[str, Any]) -> bool:
        return TSPGreedy().validate_solution(solution, problem_instance)