"""
Critical Value Simulation for the Modified KPSS Test.

This script replicates the Monte Carlo simulation methodology from
Harris, Leybourne, and McCabe (2007) for generating critical values
and power curves.

From the paper (p. 359):
"Limit distributions of the three statistics are simulated by approximating 
the Wiener process functionals involved using i.i.d.N(0,1) variables, 
approximating the integrals by normalized sums of 5,000 steps. 
All experiments are based on 10,000 replications."

Author: Dr Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nearkpss.critical_values import (
    simulate_critical_values,
    simulate_power_and_size,
    generate_critical_value_table,
)


def simulate_asymptotic_critical_values():
    """
    Simulate asymptotic critical values for the Modified KPSS test.
    
    Following the paper methodology:
    - 10,000 replications
    - 5,000 steps for Wiener process approximation
    - QS kernel for long-run variance estimation
    """
    print("=" * 70)
    print("Simulating Asymptotic Critical Values for Modified KPSS Test")
    print("Harris, Leybourne, and McCabe (2007)")
    print("=" * 70)
    print()
    
    # Parameters as in the paper
    n_replications = 10000
    n_steps = 5000
    seed = 42
    
    c_bar_values = [5, 7, 10, 13, 15, 20]
    
    print(f"Settings:")
    print(f"  Replications: {n_replications}")
    print(f"  Steps: {n_steps}")
    print(f"  Seed: {seed}")
    print()
    
    # Level case (detrend = "c")
    print("-" * 70)
    print("LEVEL CASE (Intercept only, detrend='c')")
    print("-" * 70)
    print()
    print("At the boundary c = c̄, the distribution is standard KPSS.")
    print("From Note 3: 'For S^μ(10) this is the standard KPSS value of 0.460'")
    print()
    
    print(f"{'c̄':>6} | {'1%':>10} | {'5%':>10} | {'10%':>10}")
    print("-" * 50)
    
    for c_bar in c_bar_values:
        # At boundary c = c_bar
        stats = simulate_critical_values(
            c=c_bar, c_bar=c_bar, alpha=1.0, detrend="c",
            n_replications=n_replications, n_steps=n_steps, seed=seed
        )
        cv_1 = np.percentile(stats, 99)
        cv_5 = np.percentile(stats, 95)
        cv_10 = np.percentile(stats, 90)
        
        print(f"{c_bar:>6.0f} | {cv_1:>10.4f} | {cv_5:>10.4f} | {cv_10:>10.4f}")
    
    print()
    print("Standard KPSS critical values (for comparison):")
    print("  1%:  0.739")
    print("  5%:  0.463")
    print("  10%: 0.347")
    print()
    
    # Trend case (detrend = "ct")
    print("-" * 70)
    print("TREND CASE (Intercept + Trend, detrend='ct')")
    print("-" * 70)
    print()
    
    print(f"{'c̄':>6} | {'1%':>10} | {'5%':>10} | {'10%':>10}")
    print("-" * 50)
    
    for c_bar in c_bar_values:
        stats = simulate_critical_values(
            c=c_bar, c_bar=c_bar, alpha=1.0, detrend="ct",
            n_replications=n_replications, n_steps=n_steps, seed=seed
        )
        cv_1 = np.percentile(stats, 99)
        cv_5 = np.percentile(stats, 95)
        cv_10 = np.percentile(stats, 90)
        
        print(f"{c_bar:>6.0f} | {cv_1:>10.4f} | {cv_5:>10.4f} | {cv_10:>10.4f}")
    
    print()
    print("Standard KPSS critical values (trend case):")
    print("  1%:  0.216")
    print("  5%:  0.146")
    print("  10%: 0.119")


def simulate_power_curves():
    """
    Simulate power curves as in Figure 1 of the paper.
    
    From the paper (p. 359):
    "Figure 1 shows the rejection profiles of the tests across c for a 
    rejection rate of 0.50 when c = 0."
    """
    print()
    print("=" * 70)
    print("Simulating Power Curves (Figure 1)")
    print("=" * 70)
    print()
    
    c_values = np.arange(0, 11)
    c_bar = 10.0
    nominal_power = 0.50
    n_replications = 5000  # Reduced for speed
    n_steps = 2000
    
    for alpha in [1, 2, 3]:
        print(f"\nα = {alpha}:")
        print("-" * 40)
        
        rejection_rates, cv = simulate_power_and_size(
            c_values, c_bar=c_bar, alpha=alpha, nominal_power=nominal_power,
            detrend="c", n_replications=n_replications, n_steps=n_steps, seed=42
        )
        
        print(f"{'c':>4} | {'Rejection Rate':>15}")
        print("-" * 25)
        for c, rate in zip(c_values, rejection_rates):
            print(f"{c:>4.0f} | {rate:>15.4f}")


def simulate_table1_sizes():
    """
    Replicate Table 1 from the paper: Empirical sizes at nominal 0.05 level.
    
    Settings:
    - T = 200
    - c = 10
    - v_t = ε_t - θ * ε_{t-1} (MA(1) innovations)
    - θ ∈ {0.0, 0.6, -0.6}
    - α ∈ {1, 3, 5}
    """
    print()
    print("=" * 70)
    print("Replicating Table 1: Empirical Sizes at 5% Level")
    print("=" * 70)
    print()
    print("DGP: y_t = μ + w_t, w_t = ρ_{c,T} * w_{t-1} + v_t")
    print("     v_t = ε_t - θ * ε_{t-1}, ε_t ~ i.i.d. N(0,1)")
    print("     T = 200, c = 10, c̄ = 10")
    print()
    
    # Import simulation utility
    from nearkpss.utils import simulate_near_integrated_ma
    from nearkpss import modified_kpss_test
    
    T = 200
    c = 10.0
    c_bar = 10.0
    n_sims = 1000
    nominal = 0.05
    
    theta_values = [0.0, 0.6, -0.6]
    alpha_values = [1, 3, 5]
    
    print(f"{'θ':>6} | {'α=1':>8} | {'α=3':>8} | {'α=5':>8}")
    print("-" * 42)
    
    for theta in theta_values:
        sizes = []
        for alpha in alpha_values:
            np.random.seed(42)
            rejections = 0
            
            for _ in range(n_sims):
                y = simulate_near_integrated_ma(T, c, theta, alpha, mu=0.0, sigma=1.0)
                result = modified_kpss_test(y, c_bar=c_bar, compute_pvalue=False)
                if result.reject_5pct:
                    rejections += 1
            
            size = rejections / n_sims
            sizes.append(size)
        
        print(f"{theta:>6.1f} | {sizes[0]:>8.3f} | {sizes[1]:>8.3f} | {sizes[2]:>8.3f}")
    
    print()
    print("Table 1 from paper (S^μ(10)):")
    print("       |   α=1   |   α=3   |   α=5")
    print(" θ=0.0 |  0.046  |  0.046  |  0.046")
    print(" θ=0.6 |  0.021  |  0.021  |  0.021")
    print("θ=-0.6 |  0.051  |  0.051  |  0.051")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simulate critical values and power curves for Modified KPSS test"
    )
    parser.add_argument(
        "--critical-values", action="store_true",
        help="Simulate asymptotic critical values"
    )
    parser.add_argument(
        "--power-curves", action="store_true",
        help="Simulate power curves (Figure 1)"
    )
    parser.add_argument(
        "--table1", action="store_true",
        help="Replicate Table 1 (empirical sizes)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all simulations"
    )
    
    args = parser.parse_args()
    
    if args.all or args.critical_values:
        simulate_asymptotic_critical_values()
    
    if args.all or args.power_curves:
        simulate_power_curves()
    
    if args.all or args.table1:
        simulate_table1_sizes()
    
    if not any([args.all, args.critical_values, args.power_curves, args.table1]):
        # Default: run critical values
        print("Running default: critical values simulation")
        print("Use --help for all options")
        print()
        simulate_asymptotic_critical_values()
