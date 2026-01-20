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
from scipy import stats
from itertools import combinations

# Try to import statannotations for publication-ready p-value annotations
try:
    from statannotations.Annotator import Annotator
    HAS_STATANNOTATIONS = True
except ImportError:
    HAS_STATANNOTATIONS = False
    print("Note: Install 'statannotations' for p-value annotations on plots: pip install statannotations")

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
# 1b. STATISTICAL ANALYSIS FUNCTIONS
# ==========================================

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_effect_size(d):
    """Interpret Cohen's d value."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def compute_statistical_tests(scores_dict, alpha=0.05, output_file="statistical_results.txt"):
    """
    Perform formal statistical tests on algorithm comparison results.
    
    Args:
        scores_dict: Dictionary mapping algorithm names to lists of scores
        alpha: Significance level (default 0.05)
        output_file: Path to save results (default: statistical_results.txt)
    
    Returns:
        Dictionary with test results including p-values and effect sizes
    """
    results = {
        'kruskal_wallis': None,
        'pairwise_tests': [],
        'effect_sizes': [],
        'confidence_intervals': {}
    }
    
    lines = []  # Collect output for file writing
    
    # 1. Kruskal-Wallis H-test (non-parametric ANOVA)
    groups = list(scores_dict.values())
    if len(groups) >= 2:
        stat, p_value = stats.kruskal(*groups)
        results['kruskal_wallis'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha
        }
        lines.append(f"\n--- STATISTICAL ANALYSIS (α={alpha}) ---")
        lines.append(f"Kruskal-Wallis H-test: H={stat:.4f}, p={p_value:.4e}")
        lines.append(f"  → {'Significant' if p_value < alpha else 'Not significant'} difference between groups")
    
    # 2. Pairwise Wilcoxon signed-rank tests
    algo_names = list(scores_dict.keys())
    lines.append("\nPairwise Wilcoxon signed-rank tests:")
    for (name1, name2) in combinations(algo_names, 2):
        scores1, scores2 = scores_dict[name1], scores_dict[name2]
        try:
            stat, p_value = stats.wilcoxon(scores1, scores2)
            significant = p_value < alpha
        except ValueError:
            # All differences are zero
            stat, p_value, significant = 0, 1.0, False
        
        results['pairwise_tests'].append({
            'pair': (name1, name2),
            'statistic': stat,
            'p_value': p_value,
            'significant': significant
        })
        lines.append(f"  {name1} vs {name2}: W={stat:.2f}, p={p_value:.4e} {'*' if significant else ''}")
    
    # 3. Effect sizes (Cohen's d)
    lines.append("\nEffect sizes (Cohen's d):")
    for (name1, name2) in combinations(algo_names, 2):
        d = cohens_d(scores_dict[name1], scores_dict[name2])
        interpretation = interpret_effect_size(d)
        results['effect_sizes'].append({
            'pair': (name1, name2),
            'cohens_d': d,
            'interpretation': interpretation
        })
        lines.append(f"  {name1} vs {name2}: d={d:.3f} ({interpretation})")
    
    # 4. Confidence intervals for each algorithm
    lines.append("\n95% Confidence Intervals:")
    for name, scores in scores_dict.items():
        mean = np.mean(scores)
        sem = stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=sem)
        results['confidence_intervals'][name] = {'mean': mean, 'ci_low': ci[0], 'ci_high': ci[1]}
        lines.append(f"  {name}: {mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # Print to console
    for line in lines:
        print(line)
    
    # Write to file
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Statistical Analysis Run\n")
        f.write(f"{'='*60}\n")
        for line in lines:
            f.write(line + '\n')
    print(f"\nResults appended to '{output_file}'")
    
    return results

def plot_statistical_comparison(scores_dict, output_prefix="statistical_comparison", title="Algorithm Comparison"):
    """
    Create publication-ready statistical comparison visualization.
    
    Generates:
    - Box plot with p-value annotations (if statannotations available)
    - Bar chart with 95% CI error bars  
    - Effect size heatmap
    
    Args:
        scores_dict: Dictionary mapping algorithm names to lists of scores
        output_prefix: Prefix for output files
        title: Plot title
    """
    algo_names = list(scores_dict.keys())
    
    # Prepare data for plotting
    plot_data = []
    for name, scores in scores_dict.items():
        for score in scores:
            plot_data.append({'Algorithm': name, 'Broken Edges': score})
    df = pd.DataFrame(plot_data)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Panel 1: Box plot with significance annotations ---
    ax1 = axes[0]
    sns.boxplot(data=df, x='Algorithm', y='Broken Edges', hue='Algorithm', 
                legend=False, ax=ax1, palette="viridis")
    ax1.set_title(f'{title}\n(Lower is Better)')
    ax1.set_ylim(bottom=-0.5)
    
    # Add p-value annotations if statannotations is available
    if HAS_STATANNOTATIONS and len(algo_names) >= 2:
        pairs = list(combinations(algo_names, 2))
        annotator = Annotator(ax1, pairs, data=df, x='Algorithm', y='Broken Edges')
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                           comparisons_correction=None)
        annotator.apply_and_annotate()
    
    # --- Panel 2: Mean with 95% CI error bars ---
    ax2 = axes[1]
    means = []
    ci_lows = []
    ci_highs = []
    for name in algo_names:
        scores = scores_dict[name]
        mean = np.mean(scores)
        sem = stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=sem)
        means.append(mean)
        ci_lows.append(mean - ci[0])
        ci_highs.append(ci[1] - mean)
    
    colors = sns.color_palette("viridis", len(algo_names))
    bars = ax2.bar(algo_names, means, yerr=[ci_lows, ci_highs], capsize=5, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Mean Broken Edges')
    ax2.set_title('Mean ± 95% Confidence Interval')
    ax2.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- Panel 3: Effect size heatmap ---
    ax3 = axes[2]
    n = len(algo_names)
    effect_matrix = np.zeros((n, n))
    for i, name1 in enumerate(algo_names):
        for j, name2 in enumerate(algo_names):
            if i != j:
                d = cohens_d(scores_dict[name1], scores_dict[name2])
                effect_matrix[i, j] = d
    
    # Create annotated heatmap
    mask = np.eye(n, dtype=bool)  # Mask diagonal
    sns.heatmap(effect_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                xticklabels=algo_names, yticklabels=algo_names, 
                mask=mask, ax=ax3, center=0, vmin=-2, vmax=2,
                cbar_kws={'label': "Cohen's d"})
    ax3.set_title("Effect Size (Cohen's d)\n(Row vs Column)")
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f'{output_prefix}_plot.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{fig_path}'")
    plt.close()
    
    # Save summary to CSV
    summary_data = {
        'Algorithm': algo_names,
        'Mean': means,
        'CI_Lower': [m - cl for m, cl in zip(means, ci_lows)],
        'CI_Upper': [m + ch for m, ch in zip(means, ci_highs)],
        'Std': [np.std(scores_dict[name]) for name in algo_names]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_path = f'{output_prefix}_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved '{csv_path}'")

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

def run_genetic_algorithm(graph, pop_size=200, generations=1000, cross_rate=0.8, mut_rate=0.3, operator="swap", adaptive=False):
    """
    Genetic Algorithm with selectable operator and optional adaptive mutation.
    
    When adaptive=True:
    - Calculates population diversity (unique fitness values / pop_size)
    - Low diversity (< 0.3): INCREASE mutation to escape local optima
    - High diversity (> 0.7): DECREASE mutation to exploit good solutions
    - Formula: current_mut = base_rate * (1 + (0.5 - diversity))
    - Clamped to [0.1, 0.6] range
    """
    nodes = list(graph.nodes)
    target = len(nodes) - 1
    
    mutate_func = op_inversion if operator == "inversion" else op_swap
    base_mut_rate = mut_rate

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
    
    def calculate_diversity(fitnesses):
        """Calculate diversity as ratio of unique fitness values."""
        unique = len(set(fitnesses))
        return unique / len(fitnesses)

    population = [get_initial_solution(nodes) for _ in range(pop_size)]
    best_sol = None
    best_broken = float('inf')
    trace = []
    diversity_trace = []  # Track diversity for analysis

    for gen in range(generations):
        fitnesses = [fitness_max(p, graph) for p in population]
        best_val = max(fitnesses)
        current_broken = target - best_val
        
        if current_broken < best_broken:
            best_broken = current_broken
            best_sol = population[fitnesses.index(best_val)]
        
        trace.append(best_broken)
        if current_broken == 0: break
        
        # Adaptive mutation rate adjustment
        if adaptive:
            diversity = calculate_diversity(fitnesses)
            diversity_trace.append(diversity)
            # Increase mutation when diversity is low, decrease when high
            current_mut = base_mut_rate * (1 + (0.5 - diversity))
            current_mut = max(0.1, min(0.6, current_mut))  # Clamp to [0.1, 0.6]
        else:
            current_mut = mut_rate
            
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
            if random.random() < current_mut:
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

def run_scalability_analysis(node_sizes=[30, 50, 75, 100], prob=0.15, runs=10):
    """
    Analyze how algorithm performance scales with problem size.
    
    Measures:
    - Success rate vs N
    - Mean execution time vs N
    
    Generates: experiment_4_scalability.png
    """
    print(f"\n--- 5. SCALABILITY ANALYSIS (prob={prob}, {runs} runs per N) ---")
    
    results = {'N': [], 'Success Rate': [], 'Time (s)': [], 'Algorithm': []}
    
    algorithms = {
        'SA': lambda g: run_simulated_annealing(g, max_steps=3000, operator="inversion"),
        'Tabu': lambda g: run_tabu_search(g, max_steps=1500, operator="inversion"),
        'GA': lambda g: run_genetic_algorithm(g, generations=1500, operator="inversion")
    }
    
    for n in node_sizes:
        print(f"Testing N={n}...")
        # Create graph with probability high enough to likely have a solution
        g = nx.erdos_renyi_graph(n=n, p=prob, seed=SEED)
        
        for name, algo_func in algorithms.items():
            successes = 0
            times = []
            for _ in range(runs):
                start = time.time()
                _, cost, _ = algo_func(g)
                dur = time.time() - start
                times.append(dur)
                if cost == 0:
                    successes += 1
            
            rate = (successes / runs) * 100
            mean_time = np.mean(times)
            results['N'].append(n)
            results['Success Rate'].append(rate)
            results['Time (s)'].append(mean_time)
            results['Algorithm'].append(name)
            print(f"  {name}: Success={rate:.0f}%, Time={mean_time:.3f}s")
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.lineplot(data=df, x='N', y='Success Rate', hue='Algorithm', marker='o', ax=axes[0], linewidth=2)
    axes[0].set_title('Success Rate vs Problem Size')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_xlabel('Number of Nodes (N)')
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True)
    
    sns.lineplot(data=df, x='N', y='Time (s)', hue='Algorithm', marker='o', ax=axes[1], linewidth=2)
    axes[1].set_title('Execution Time vs Problem Size')
    axes[1].set_ylabel('Mean Time (s)')
    axes[1].set_xlabel('Number of Nodes (N)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_4_scalability.png', dpi=300, bbox_inches='tight')
    print("Saved 'experiment_4_scalability.png'")
    
    # Save results to CSV and text file
    df.to_csv('experiment_4_scalability_results.csv', index=False)
    print("Saved 'experiment_4_scalability_results.csv'")
    
    with open('statistical_results.txt', 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Scalability Analysis (prob={prob}, {runs} runs per N)\n")
        f.write(f"{'='*60}\n")
        for n in node_sizes:
            f.write(f"\nN={n}:\n")
            for algo in ['SA', 'Tabu', 'GA']:
                row = df[(df['N'] == n) & (df['Algorithm'] == algo)]
                if not row.empty:
                    f.write(f"  {algo}: Success={row['Success Rate'].values[0]:.0f}%, Time={row['Time (s)'].values[0]:.3f}s\n")
    print("Results appended to 'statistical_results.txt'")

def run_adaptive_comparison(num_nodes, prob=0.1, runs=20):
    """
    Compare standard GA vs adaptive mutation GA.
    
    Generates: experiment_5_adaptive_comparison.png
    """
    print(f"\n--- 6. ADAPTIVE MUTATION COMPARISON (N={num_nodes}, p={prob}) ---")
    
    g = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=SEED)
    
    # Run standard GA
    print("Running Standard GA (fixed mutation)...")
    standard_scores = []
    standard_traces = []
    for _ in range(runs):
        _, cost, trace = run_genetic_algorithm(g, generations=1000, operator="inversion", adaptive=False)
        standard_scores.append(cost)
        standard_traces.append(trace)
    
    # Run adaptive GA
    print("Running Adaptive GA (diversity-based mutation)...")
    adaptive_scores = []
    adaptive_traces = []
    for _ in range(runs):
        _, cost, trace = run_genetic_algorithm(g, generations=1000, operator="inversion", adaptive=True)
        adaptive_scores.append(cost)
        adaptive_traces.append(trace)
    
    # Print summary
    print(f"\nStandard GA: Mean={np.mean(standard_scores):.2f}, Success={(standard_scores.count(0)/runs)*100:.0f}%")
    print(f"Adaptive GA: Mean={np.mean(adaptive_scores):.2f}, Success={(adaptive_scores.count(0)/runs)*100:.0f}%")
    
    # Statistical test and visualization
    scores_dict = {'Standard GA': standard_scores, 'Adaptive GA': adaptive_scores}
    compute_statistical_tests(scores_dict, output_file="adaptive_comparison_stats.txt")
    plot_statistical_comparison(scores_dict, 
                                output_prefix="adaptive_comparison_stats",
                                title=f"Standard vs Adaptive GA (N={num_nodes})")
    
    # Plot convergence comparison (using median trace)
    def get_median_trace(traces):
        max_len = max(len(t) for t in traces)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
        return np.median(padded, axis=0)
    
    median_standard = get_median_trace(standard_traces)
    median_adaptive = get_median_trace(adaptive_traces)
    
    plt.figure(figsize=(10, 6))
    plt.plot(median_standard, label='Standard GA (Fixed Mutation)', linewidth=2)
    plt.plot(median_adaptive, label='Adaptive GA (Diversity-Based)', linewidth=2, linestyle='--')
    plt.title(f'GA Convergence: Standard vs Adaptive Mutation (N={num_nodes})')
    plt.xlabel('Generation')
    plt.ylabel('Broken Edges (Cost)')
    plt.ylim(bottom=-0.5)
    plt.legend()
    plt.grid(True)
    
    plt.savefig('experiment_5_adaptive_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 'experiment_5_adaptive_comparison.png'")
    
    # Save detailed results to file
    with open('statistical_results.txt', 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Adaptive Mutation Comparison (N={num_nodes}, p={prob})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Standard GA: Mean={np.mean(standard_scores):.2f}, Success={(standard_scores.count(0)/runs)*100:.0f}%\n")
        f.write(f"Adaptive GA: Mean={np.mean(adaptive_scores):.2f}, Success={(adaptive_scores.count(0)/runs)*100:.0f}%\n")
    print("Results appended to 'statistical_results.txt'")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hamiltonian Path Metaheuristic Analysis")
    parser.add_argument("-N", "--nodes", type=int, default=50, help="Number of nodes in the graph (default: 50)")
    parser.add_argument("-p", "--prob", type=float, default=0.1, help="Edge creation probability (default: 0.1)")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "batch", "opt", "phase", "visual", "stats", "scale", "adaptive"],
                        help="Experiment mode to run individually")

    
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
    
    # 5. Statistical Analysis Mode (batch + stats)
    if args.mode == "stats":
        print("\n--- RUNNING BATCH WITH STATISTICAL ANALYSIS ---")
        OPERATOR = "inversion"
        g_exp = nx.erdos_renyi_graph(n=args.nodes, p=args.prob, seed=SEED)
        
        def run_batch_for_stats(algo_func, name):
            scores = []
            for _ in range(30):
                _, cost, _ = algo_func(g_exp)
                scores.append(cost)
            return scores
        
        sa_scores = run_batch_for_stats(lambda g: run_simulated_annealing(g, max_steps=3000, operator=OPERATOR), "SA")
        tabu_scores = run_batch_for_stats(lambda g: run_tabu_search(g, max_steps=1500, operator=OPERATOR), "Tabu")
        ga_scores = run_batch_for_stats(lambda g: run_genetic_algorithm(g, generations=1500, operator=OPERATOR), "GA")
        
        scores_dict = {'SA': sa_scores, 'Tabu': tabu_scores, 'GA': ga_scores}
        compute_statistical_tests(scores_dict)
        plot_statistical_comparison(scores_dict, 
                                    output_prefix=f"stats_N{args.nodes}_p{args.prob}",
                                    title=f"Algorithm Comparison (N={args.nodes}, p={args.prob})")
    
    # 6. Scalability Analysis
    if args.mode == "scale":
        run_scalability_analysis(node_sizes=[30, 50, 75, 100], prob=0.15)
    
    # 7. Adaptive Mutation Comparison
    if args.mode == "adaptive":
        run_adaptive_comparison(args.nodes, prob=args.prob)