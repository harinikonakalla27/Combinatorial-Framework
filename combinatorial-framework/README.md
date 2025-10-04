# Combinatorial Optimization Framework

A unified framework for solving combinatorial optimization problems using multiple strategies including:
- Greedy Algorithms
- Dynamic Programming
- Backtracking
- Branch and Bound
- Divide and Conquer

## Problems Implemented
- Traveling Salesman Problem (TSP)
- Knapsack Problem
- Graph Matching

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the Django database:
```bash
python manage.py migrate
```

5. Run the development server:
```bash
python manage.py runserver
```

## Project Structure

```
combi_opt/
├── algorithms/           # Core algorithm implementations
│   ├── greedy/
│   ├── dynamic/
│   ├── backtrack/
│   ├── branch_bound/
│   └── divide_conquer/
├── problems/            # Problem-specific implementations
│   ├── tsp/
│   ├── knapsack/
│   └── graph_matching/
├── visualization/       # Visualization utilities
├── benchmarks/         # Performance testing
└── web_interface/      # Django web interface
```

## Usage

Visit http://localhost:8000 after starting the server to access the web interface.

## Testing

Run tests using:
```bash
pytest
```