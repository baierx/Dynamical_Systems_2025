import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# def model(t, variables, a1, b1, a2, b2, k_max, K_m, k_i, n, m, q):
#     """Coupled system with feedback inhibition"""
#     S, P = variables
    
#     enzymatic_rate = (k_max * S**n) / (K_m**m + S**m) / (1 + k_i * P**q)
    
#     dSdt = h_ex - b1 * S - enzymatic_rate
#     dPdt = a2 - b2 * P + enzymatic_rate
    
#     return [dSdt, dPdt]

def dX_dt(t, X, h_ex):
    """Return the rates at all positions."""
    h_in, tau_ex, tau_in, c2, c4, c_EE, c_EI = (-4., 1, 1.5, 10, 0, 5, 10)

    
    return np.array([
        (h_ex - X[0] - c2*np.tanh(X[1]) + c_EE*np.tanh(X[0]))*tau_ex,
        (h_in - X[1] - c4*np.tanh(X[1]) + c_EI*np.tanh(X[0]))*tau_in
    ])



def find_asymptotic_extrema(h_ex_value, model_func, initial_conditions, 
                            t_transient=500, t_analysis=200):
    """
    Simulate the system and extract min/max values in asymptotic regime.
    
    This simplified version just takes the min/max of the time series,
    which is more robust than peak detection for bifurcation diagrams.
    """
    t_total = t_transient + t_analysis
        
    # Solve ODE
    sol = solve_ivp(model_func, [0, t_total], initial_conditions, 
                    args=(h_ex_value,), method='LSODA', dense_output=True,
                    rtol=1e-8, atol=1e-10)
    
    # Evaluate in asymptotic regime
    t_eval = np.linspace(t_transient, t_total, 200)
    solution = sol.sol(t_eval)
    
    Ex_asymptotic = solution[0, :]
    In_asymptotic = solution[1, :]
    
    # Simply take min and max of the time series
    Ex_min = np.min(Ex_asymptotic)
    Ex_max = np.max(Ex_asymptotic)
    In_min = np.min(In_asymptotic)
    In_max = np.max(In_asymptotic)
    
    return Ex_min, Ex_max, In_min, In_max


def create_bifurcation_diagram(h_ex_range, model_func, initial_conditions, n_points=100, t_transient=500,
                               t_analysis=200):
    """Create bifurcation diagram data."""
    
    h_ex_values = np.linspace(h_ex_range[0], h_ex_range[1], n_points)
    
    Ex_mins = []
    Ex_maxs = []
    In_mins = []
    In_maxs = []
    
    # Use continuation
    current_ic = initial_conditions.copy()
    
    print("Computing bifurcation diagram...")
    print(f"Scanning h_ex from {h_ex_range[0]:.4f} to {h_ex_range[1]:.4f}")
    print(f"Number of points: {n_points}")
    print("-" * 60)
    
    for i, h_ex in enumerate(h_ex_values):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Progress: {i+1:3d}/{n_points} | h_ex = {h_ex:.6f}")
        
        try:
            Ex_min, Ex_max, In_min, In_max = find_asymptotic_extrema(
                h_ex, model_func, current_ic,
                t_transient=t_transient, t_analysis=t_analysis
            )
            
            Ex_mins.append(Ex_min)
            Ex_maxs.append(Ex_max)
            In_mins.append(In_min)
            In_maxs.append(In_max)
            
            # Update initial conditions for continuation
            current_ic = [(Ex_min + Ex_max)/2, (In_min + In_max)/2]
            
        except Exception as e:
            print(f"Warning: Failed at h_ex = {h_ex:.6f}: {e}")
            # Use previous values
            if len(Ex_mins) > 0:
                Ex_mins.append(Ex_mins[-1])
                Ex_maxs.append(Ex_maxs[-1])
                In_mins.append(In_mins[-1])
                In_maxs.append(In_maxs[-1])
            else:
                Ex_mins.append(np.nan)
                Ex_maxs.append(np.nan)
                In_mins.append(np.nan)
                In_maxs.append(np.nan)
    
    print("-" * 60)
    print("Bifurcation diagram computation complete!")
    
    return (np.array(h_ex_values), 
            np.array(Ex_mins), np.array(Ex_maxs),
            np.array(In_mins), np.array(In_maxs))


def plot_bifurcation_diagram(h_ex_values, Ex_mins, Ex_maxs, In_mins, In_maxs, 
                             save_filename='bifurcation_diagram_corrected.png',
                             oscillation_range=None):
    """
    Plot the bifurcation diagram with BOTH min and max clearly visible.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Highlight oscillation region if provided
    if oscillation_range is not None:
        ax1.axvspan(oscillation_range[0], oscillation_range[1], 
                   alpha=0.15, color='yellow', label='Expected oscillations')
        ax2.axvspan(oscillation_range[0], oscillation_range[1], 
                   alpha=0.15, color='yellow', label='Expected oscillations')
    
    # Plot Ex bifurcation diagram
    # Plot minima in BLUE and maxima in RED
    ax1.plot(h_ex_values, Ex_mins, 'o', color='blue', markersize=4, 
             alpha=0.6, label='S min', markeredgewidth=0)
    ax1.plot(h_ex_values, Ex_maxs, 'o', color='red', markersize=4, 
             alpha=0.6, label='S max', markeredgewidth=0)
    
    ax1.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Ex', fontsize=13, fontweight='bold')
    ax1.set_title('Bifurcation Diagram - Ex', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.tick_params(labelsize=11)
    
    # Plot product bifurcation diagram
    ax2.plot(h_ex_values, In_mins, 'o', color='blue', markersize=4, 
             alpha=0.6, label='In min', markeredgewidth=0)
    ax2.plot(h_ex_values, In_maxs, 'o', color='red', markersize=4, 
             alpha=0.6, label='In max', markeredgewidth=0)
    
    ax2.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
    ax2.set_ylabel('In', fontsize=13, fontweight='bold')
    ax2.set_title('Bifurcation Diagram - In', 
                  fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.tick_params(labelsize=11)
    
    plt.tight_layout()
    # plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    # print(f"\nFigure saved as: {save_filename}")
    
    return fig


def analyze_oscillations(h_ex_values, Ex_mins, Ex_maxs, In_mins, In_maxs):
    """Analyze where oscillations occur."""
    
    Ex_amplitude = Ex_maxs - Ex_mins
    In_amplitude = In_maxs - In_mins
    
    # Threshold for considering it an oscillation
    threshold = 0.01
    
    Ex_oscillating = Ex_amplitude > threshold
    In_oscillating = In_amplitude > threshold
    
    print("\n" + "="*70)
    print("OSCILLATION ANALYSIS")
    print("="*70)
    
    # Find oscillatory regions
    if np.any(Ex_oscillating):
        osc_indices = np.where(Ex_oscillating)[0]
        h_ex_osc_start = h_ex_values[osc_indices[0]]
        h_ex_osc_end   = h_ex_values[osc_indices[-1]]
        print(f"\Ex oscillations detected:")
        print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(Ex_amplitude)
        print(f"  Maximum amplitude: {Ex_amplitude[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
    else:
        print("\nNo Ex oscillations detected (all steady state)")
    
    if np.any(In_oscillating):
        osc_indices = np.where(In_oscillating)[0]
        h_ex_osc_start = h_ex_values[osc_indices[0]]
        h_ex_osc_end = h_ex_values[osc_indices[-1]]
        print(f"\nProduct oscillations detected:")
        print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(In_amplitude)
        print(f"  Maximum amplitude: {In_amplitude[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
    else:
        print("\nNo product oscillations detected (all steady state)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("BIFURCATION ANALYSIS")
    print("E-I Neural Population Model")
    print("="*70)
    
    # Model parameters
    Ex_0 = 1
    In_0 = -1
    initial_conditions = [Ex_0, In_0]
    
    # Bifurcation parameter range
    h_ex_min = -8
    h_ex_max =  8.0
    n_points = 150
    
    # Simulation parameters
    t_transient = 50
    t_analysis = 50
    
    print(f"\nParameters:")
    print(f"  h_ex range: [{h_ex_min}, {h_ex_max}]")
    print(f"  Points: {n_points}")
    print(f"  Initial conditions: Ex_0={Ex_0}, In_0={In_0}")
    print()
    
    # Create bifurcation diagram
    h_ex_vals, Ex_mins, Ex_maxs, In_mins, In_maxs = create_bifurcation_diagram(
        [h_ex_min, h_ex_max],
        dX_dt,
        initial_conditions,
        n_points=n_points,
        t_transient=t_transient,
        t_analysis=t_analysis
    )
    
    # Plot
    plot_bifurcation_diagram(h_ex_vals, Ex_mins, Ex_maxs, In_mins, In_maxs,
                            save_filename='EI_bifurcation_diagram.png',
                            oscillation_range=[-6.95, 6.8])
    
    # Analyze
    analyze_oscillations(h_ex_vals, Ex_mins, Ex_maxs, In_mins, In_maxs)
    
    # Print some sample values
    print("\nSample values:")
    print("h_ex      | Ex_min  | Ex_max  | In_min  | In_max  ")
    print("-" * 55)
    for i in [0, 10, 50, 100, 149]:
        print(f"{h_ex_vals[i]:.4f} | {Ex_mins[i]:.4f} | {Ex_maxs[i]:.4f} | {In_mins[i]:.4f} | {In_maxs[i]:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    plt.show()
