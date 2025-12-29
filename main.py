import random
import time
import math
import itertools
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp, log

# --- CONFIGURATION ---
SNS_THEME = "whitegrid"
SNS_CONTEXT = "paper"
SEED = 42

# Apply settings
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style=SNS_THEME, context=SNS_CONTEXT, font_scale=1.2)

# ==========================================
# 1. CORE HELPER & FITNESS FUNCTIONS
# ==========================================

def get_initial_solution(nodes):
    """Returns a random permutation of nodes."""
    p = list(nodes).copy()
    random.shuffle(p)
    return p

def count_edges(path, graph):
    """Counts valid edges in the path for a specific graph."""
    edges = 0
    # Optimization: Iterate via index to avoid creating new lists
    for i in range(len(path) - 1):
        if graph.has_edge(path[i], path[i+1]):
            edges += 1
    return edges

def fitness_min(path, graph):
    """Minimisation: Returns number of broken edges. Target = 0."""
    target = len(path) - 1
    valid = count_edges(path, graph)
    return float(target - valid)

def fitness_max(path, graph):
    """Maximisation: Returns number of valid edges. Target = N-1."""
    return float(count_edges(path, graph))

# ==========================================
# 2. OPERATORS (THE OPTIMISATIONS)
# ==========================================

def op_swap(genome):
    """Standard Swap: Exchanges two random nodes."""
    n = len(genome)
    i, j = random.sample(range(n), 2)
    genome[i], genome[j] = genome[j], genome[i]
    return genome

def op_inversion(genome):
    """
    2-Opt Inversion: Reverses a random sub-segment.
    OPTIMIZATION: Preserves adjacency better than swap.
    """
    n = len(genome)
    i, j = sorted(random.sample(range(n), 2))
    # Reverse the segment between i and j
    genome[i:j+1] = genome[i:j+1][::-1]
    return genome

# ==========================================
# 3. METAHEURISTIC ALGORITHMS
# ==========================================

def run_simulated_annealing(graph, max_steps=3000, temp0=100.0, operator="swap"):
    """Simulated Annealing with selectable operator."""
    nodes = list(graph.nodes)
    current = get_initial_solution(nodes)
    best = current.copy()
    
    current_cost = fitness_min(current, graph)
    best_cost = current_cost
    trace = [best_cost]
    temp = temp0
    
    mutate_func = op_inversion if operator == "inversion" else op_swap
    
    for step in range(max_steps):
        if best_cost == 0: break
            
        neighbor = current.copy()
        neighbor = mutate_func(neighbor)
        
        neighbor_cost = fitness_min(neighbor, graph)
        delta = neighbor_cost - current_cost
        
        if delta < 0 or random.random() < exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        
        trace.append(best_cost)
        temp *= 0.985 
        
    return best, best_cost, trace

def run_tabu_search(graph, max_steps=1000, tenure=20, operator="swap"):
    """Tabu Search with selectable operator."""
    nodes = list(graph.nodes)
    current = get_initial_solution(nodes)
    best = current.copy()
    best_cost = fitness_min(best, graph)
    
    tabu_list = []
    trace = [best_cost]
    mutate_func = op_inversion if operator == "inversion" else op_swap
    
    for step in range(max_steps):
        if best_cost == 0: break
            
        candidates = []
        for _ in range(50):
            cand = current.copy()
            cand = mutate_func(cand)
            candidates.append(cand)
            
        candidates.sort(key=lambda p: fitness_min(p, graph))
        
        found_move = False
        for cand in candidates:
            cand_cost = fitness_min(cand, graph)
            if cand_cost < best_cost:
                current = cand
                best = cand
                best_cost = cand_cost
                found_move = True
                break
            if cand not in tabu_list:
                current = cand
                found_move = True
                break
        
        trace.append(best_cost)
        if found_move:
            tabu_list.append(current)
            if len(tabu_list) > tenure:
                tabu_list.pop(0)

    return best, best_cost, trace

def run_genetic_algorithm(graph, pop_size=200, generations=1000, cross_rate=0.8, mut_rate=0.3, operator="swap"):
    """Genetic Algorithm with selectable operator."""
    nodes = list(graph.nodes)
    target = len(nodes) - 1
    
    mutate_func = op_inversion if operator == "inversion" else op_swap

    def ordered_crossover(p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = p1[start:end]
        current_p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[current_p2_idx] in child:
                    current_p2_idx += 1
                child[i] = p2[current_p2_idx]
        return child

    population = [get_initial_solution(nodes) for _ in range(pop_size)]
    best_sol = None
    best_broken = float('inf')
    trace = [] 

    for gen in range(generations):
        fitnesses = [fitness_max(p, graph) for p in population]
        best_val = max(fitnesses)
        current_broken = target - best_val
        
        if current_broken < best_broken:
            best_broken = current_broken
            best_sol = population[fitnesses.index(best_val)]
        
        trace.append(best_broken)
        if current_broken == 0: break
            
        next_pop = [best_sol.copy()] 
        while len(next_pop) < pop_size:
            competitors = random.sample(population, 3)
            winner = max(competitors, key=lambda p: fitness_max(p, graph))
            next_pop.append(winner.copy())
        population = next_pop

        for i in range(1, pop_size - 1, 2):
            if random.random() < cross_rate:
                population[i] = ordered_crossover(population[i], population[i+1])
                
        for i in range(1, pop_size):
            if random.random() < mut_rate:
                population[i] = mutate_func(population[i])

    return best_sol, best_broken, trace

# ==========================================
# 4. EXPERIMENT RUNNERS
# ==========================================

def run_batch_experiments(num_nodes, prob):
    """Runs each algorithm 30 times and plots results."""
    # Baseline comparison using Standard Swap
    OPERATOR = "swap" 
    
    g_exp = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=SEED)
    
    print(f"\n--- 1. BATCH EXPERIMENT (Baseline 'Swap', N={num_nodes}, p={prob}, 30 Runs) ---")
    
    def run_batch(algo_func, name):
        scores = []
        times = []
        successes = 0
        for _ in range(30):
            start = time.time()
            _, cost, _ = algo_func(g_exp)
            dur = time.time() - start
            scores.append(cost)
            times.append(dur)
            if cost == 0: successes += 1
        
        rate = (successes/30)*100
        print(f"{name}: Mean Cost={np.mean(scores):.2f}, Success={rate:.1f}%, Mean Time={np.mean(times):.4f}s")
        return scores, times

    sa_scores, sa_times = run_batch(lambda g: run_simulated_annealing(g, max_steps=3000, operator=OPERATOR), "SA")
    tabu_scores, tabu_times = run_batch(lambda g: run_tabu_search(g, max_steps=1500, operator=OPERATOR), "Tabu")
    ga_scores, ga_times = run_batch(lambda g: run_genetic_algorithm(g, generations=1500, operator=OPERATOR), "GA")

    data = {
        'Algorithm': ['SA']*30 + ['Tabu']*30 + ['GA']*30,
        'Broken Edges': sa_scores + tabu_scores + ga_scores,
        'Time (s)': sa_times + tabu_times + ga_times
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df, x='Algorithm', y='Broken Edges', hue='Algorithm', 
                legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title(f'Solution Quality (N={num_nodes}, Lower is Better)')
    axes[0].set_ylim(bottom=-0.5) 
    
    sns.boxplot(data=df, x='Algorithm', y='Time (s)', hue='Algorithm', 
                legend=False, ax=axes[1], palette="magma")
    axes[1].set_title(f'Execution Time (N={num_nodes})')
    axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('experiment_1_batch_results.png', dpi=300, bbox_inches='tight')
    print("Saved 'experiment_1_batch_results.png'")
    # plt.show() # Uncomment if running interactively

def run_optimization_comparison(num_nodes):
    """Compares Standard 'Swap' vs Optimized 'Inversion' (2-Opt)."""
    print(f"\n--- 2. OPTIMISATION EVALUATION (N={num_nodes}) ---")
    g_opt = nx.erdos_renyi_graph(n=num_nodes, p=0.1, seed=SEED)
    
    print("Running SA with Standard Swap...")
    _, _, trace_swap = run_simulated_annealing(g_opt, max_steps=4000, operator="swap")
    
    print("Running SA with Optimized Inversion (2-Opt)...")
    _, _, trace_inv = run_simulated_annealing(g_opt, max_steps=4000, operator="inversion")
    
    df_opt = pd.DataFrame({
        'Step': list(range(len(trace_swap))) + list(range(len(trace_inv))),
        'Broken Edges': trace_swap + trace_inv,
        'Variant': ['Standard (Swap)'] * len(trace_swap) + ['Optimized (2-Opt)'] * len(trace_inv)
    })
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_opt, x='Step', y='Broken Edges', hue='Variant', linewidth=2.5)
    plt.title(f'Impact of Neighborhood Structure: Swap vs 2-Opt (N={num_nodes})')
    plt.ylabel('Broken Edges (Cost)')
    plt.xlabel('Iteration Step')
    plt.ylim(bottom=-0.5)
    
    plt.savefig('experiment_2_optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 'experiment_2_optimization_comparison.png'")

def run_phase_transition(num_nodes):
    """
    Analyzes difficulty vs graph density across ALL algorithms.
    This provides a comprehensive view of algorithmic limits.
    """
    print(f"\n--- 3. PHASE TRANSITION ANALYSIS (N={num_nodes}) ---")
    # Densities to test
    densities = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
    
    # Calculate Theoretical Threshold for THIS N
    # Komlós & Szemerédi theorem: p = (ln(n) + ln(ln(n))) / n
    if num_nodes > 1:
        math_threshold = (log(num_nodes) + log(log(num_nodes))) / num_nodes
        print(f"Theoretical Critical Threshold for N={num_nodes}: p ~ {math_threshold:.3f}")
    else:
        math_threshold = 0
    
    # Store results for plotting
    results = {'Density': [], 'Success Rate': [], 'Algorithm': []}
    
    # Test all algorithms to see which one handles the transition best
    algorithms = {
        'SA': lambda g: run_simulated_annealing(g, max_steps=2500, operator="inversion"),
        'Tabu': lambda g: run_tabu_search(g, max_steps=1500, operator="inversion"),
        'GA': lambda g: run_genetic_algorithm(g, generations=1000, operator="inversion")
    }

    for p in densities:
        # Create a graph for this density
        # Note: Ideally we average over multiple graphs, but for speed we use one seed per density
        # or we generate a fresh one each run. Let's stick to one graph instance per density 
        # but 10 runs per algorithm on it.
        g = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=999)
        
        print(f"Testing Density p={p}...")
        
        for name, algo_func in algorithms.items():
            successes = 0
            for _ in range(10): # 10 runs per density/algorithm pair
                _, cost, _ = algo_func(g)
                if cost == 0: successes += 1
            
            rate = (successes/10)*100
            results['Density'].append(p)
            results['Success Rate'].append(rate)
            results['Algorithm'].append(name)
            
    df_phase = pd.DataFrame(results)
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_phase, x='Density', y='Success Rate', hue='Algorithm', marker='o', linewidth=2)
    
    # Plot the Theoretical Line
    plt.axvline(x=math_threshold, color='r', linestyle='--', label=f'Theoretical Limit (p={math_threshold:.2f})')
    
    plt.title(f'Phase Transition: Experiment vs Theory (N={num_nodes})')
    plt.xlabel('Graph Density (p)')
    plt.ylabel('Success Rate (%)')
    plt.ylim(-5, 105)
    plt.legend()
    plt.grid(True)
    
    plt.savefig('experiment_3_phase_transition.png', dpi=300, bbox_inches='tight')
    print("Saved 'experiment_3_phase_transition.png'")

def save_best_graph_html(num_nodes, prob):
    """Runs GA once and saves the result to HTML."""
    print("\n--- 4. GENERATING VISUALISATION ---")
    try:
        from pyvis.network import Network
    except ImportError:
        print("Pyvis not installed. Skipping.")
        return

    # Use arguments, but if probability is too low, we might not find a path
    g = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=SEED)
    path, cost, _ = run_genetic_algorithm(g, generations=2000, operator="inversion")
    
    print(f"Final Path Cost: {cost}")
    
    net = Network(height="600px", width="100%", cdn_resources='remote')
    
    for n in g.nodes:
        net.add_node(int(n), label=str(n), color='#97c2fc')
        
    for u, v in g.edges:
        net.add_edge(int(u), int(v), color='#e0e0e0', width=1)
        
    for i in range(len(path) - 1):
        u, v = int(path[i]), int(path[i+1])
        if g.has_edge(u, v):
            net.add_edge(u, v, color='red', width=4)
        else:
            net.add_edge(u, v, color='red', width=4, dashes=True) 
            
    net.show("hamiltonian_path.html", notebook=False)
    print("Saved to 'hamiltonian_path.html'")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hamiltonian Path Metaheuristic Analysis")
    parser.add_argument("-N", "--nodes", type=int, default=50, help="Number of nodes in the graph (default: 50)")
    parser.add_argument("-p", "--prob", type=float, default=0.1, help="Edge creation probability (default: 0.1)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "batch", "opt", "phase", "visual"], help="Experiment mode to run individually")

    
    args = parser.parse_args()
    
    print(f"Running experiments with N={args.nodes} and p={args.prob}, Mode={args.mode}")
    
    # 1. Main Baseline (using 'Swap')
    if args.mode in ["all", "batch"]:
        run_batch_experiments(args.nodes, args.prob)
    
    # 2. The Report Recommendation (Proving 2-Opt is better)
    if args.mode in ["all", "opt"]:
        run_optimization_comparison(args.nodes)
    
    # 3. Physics/Difficulty Analysis (Runs ALL algorithms across densities)
    if args.mode in ["all", "phase"]:
        run_phase_transition(args.nodes)
    
    # 4. Visual Output
    if args.mode in ["all", "visual"]:
        save_best_graph_html(args.nodes, args.prob)