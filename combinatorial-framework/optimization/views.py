from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import time
import traceback

from .algorithms.tsp_solver import TSPGreedy, TSPDynamic, TSPBacktracking
from .algorithms.knapsack_solver import KnapsackGreedy, KnapsackDynamic, KnapsackBacktracking

def index(request):
    """Render the main application page."""
    return render(request, 'optimization/index.html')

@csrf_exempt
def solve_tsp(request):
    """Handle TSP problem solving requests."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    
    try:
        data = json.loads(request.body)
        if 'distances' in data:
            distances = np.array(data.get('distances', []))
        elif 'coordinates' in data:
            coords = np.array(data['coordinates'])
            n = len(coords)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        else:
            return JsonResponse({'error': 'Either distance matrix or coordinates are required'}, status=400)
        
        strategy = data.get('strategy', 'greedy')
        
        if distances.size == 0:
            return JsonResponse({'error': 'Invalid input data'}, status=400)
        
        problem_instance = {'distances': distances}
        
        # Time the solution
        start_time = time.time()
        
        if strategy == 'greedy':
            solver = TSPGreedy()
        elif strategy == 'dynamic':
            if len(distances) > 15:  # Dynamic programming is exponential
                return JsonResponse({
                    'error': 'Dynamic programming strategy is not suitable for problems with more than 15 cities'
                }, status=400)
            solver = TSPDynamic()
        elif strategy == 'backtrack':
            if len(distances) > 20:  # Backtracking is factorial time
                return JsonResponse({
                    'error': 'Backtracking strategy is not suitable for problems with more than 20 cities'
                }, status=400)
            solver = TSPBacktracking()
        else:
            return JsonResponse({'error': 'Invalid strategy'}, status=400)
        
        try:
            solution = solver.solve(problem_instance)
            runtime = time.time() - start_time
            
            if not solver.validate_solution(solution, problem_instance):
                return JsonResponse({'error': 'Invalid solution produced'}, status=500)
            
            return JsonResponse({
                'path': solution['path'],
                'distance': float(solution['distance']),
                'runtime': runtime,
                'strategy': strategy
            })
        except MemoryError:
            return JsonResponse({
                'error': 'Problem too large for selected strategy'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'error': f'Solver error: {str(e)}\n{traceback.format_exc()}'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({'error': f'Request error: {str(e)}'}, status=400)

@csrf_exempt
def solve_knapsack(request):
    """Handle Knapsack problem solving requests."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    
    try:
        data = json.loads(request.body)
        weights = np.array(data.get('weights', []), dtype=float)
        values = np.array(data.get('values', []), dtype=float)
        capacity = float(data.get('capacity', 0))
        strategy = data.get('strategy', 'greedy')
        
        if len(weights) == 0 or len(values) == 0 or capacity <= 0:
            return JsonResponse({'error': 'Valid weights, values, and capacity are required'}, status=400)
        
        if len(weights) != len(values):
            return JsonResponse({'error': 'Number of weights must match number of values'}, status=400)
        
        problem_instance = {
            'weights': weights,
            'values': values,
            'capacity': capacity
        }
        
        start_time = time.time()
        
        if strategy == 'greedy':
            solver = KnapsackGreedy()
        elif strategy == 'dynamic':
            if len(weights) > 1000 or capacity > 10000:  # Limit problem size for dynamic programming
                return JsonResponse({
                    'error': 'Dynamic programming strategy is not suitable for large problems'
                }, status=400)
            solver = KnapsackDynamic()
        elif strategy == 'backtrack':
            if len(weights) > 30:  # Backtracking is exponential
                return JsonResponse({
                    'error': 'Backtracking strategy is not suitable for problems with more than 30 items'
                }, status=400)
            solver = KnapsackBacktracking()
        else:
            return JsonResponse({'error': 'Invalid strategy'}, status=400)
        
        try:
            solution = solver.solve(problem_instance)
            runtime = time.time() - start_time
            
            if not solver.validate_solution(solution, problem_instance):
                return JsonResponse({'error': 'Invalid solution produced'}, status=500)
            
            return JsonResponse({
                'selected_items': solution['selected_items'],
                'total_value': float(solution['total_value']),
                'total_weight': float(solution['total_weight']),
                'runtime': runtime,
                'strategy': strategy
            })
        except MemoryError:
            return JsonResponse({
                'error': 'Problem too large for selected strategy'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'error': f'Solver error: {str(e)}\n{traceback.format_exc()}'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
