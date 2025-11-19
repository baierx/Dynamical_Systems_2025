import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from numpy import tanh as sigmoid


def dX_dt(t, X, h_ex):
    """Return the rates at all positions."""
    h_in_1, h_in_2, tau_ex, tau_in1, tau_in2, c1, c2, c3, c4, c5, c6, c7 = (
        -4.5, 0.5, 1, 2.0, 0.01, 12, 10, 10, 2, 5, 5, 3)

    
    return np.array([
        (h_ex   - X[0] + c1*sigmoid(X[0]) - c2*sigmoid(X[1]) - c7*sigmoid(X[2])) * tau_ex,
        (h_in_1 - X[1] + c3*sigmoid(X[0]) - c4*sigmoid(X[1]))                    * tau_in1, 
        (h_in_2 - X[2] + c5*sigmoid(X[0]) - c6*sigmoid(X[2]))                    * tau_in2
    ])



# def find_asymptotic_extrema(h_ex_value, model_func, initial_conditions,
#                            t_transient=500, t_analysis=200):
#     """
#     Simulate the system and extract min/max values in asymptotic regime.
#     """
#     t_total = t_transient + t_analysis
    
#     # Solve ODE
#     sol = solve_ivp(model_func, [0, t_total], initial_conditions, 
#                     args=(h_ex_value,), method='LSODA', dense_output=True,
#                     rtol=1e-8, atol=1e-10)

#     # Evaluate in asymptotic regime
#     t_eval = np.linspace(t_transient, t_total, 200)
#     solution = sol.sol(t_eval)

#     Ex_asymptotic   = solution[0, :]
#     In_1_asymptotic = solution[1, :]
#     In_2_asymptotic = solution[2, :]  # ADD THIS

#     # Simply take min and max of the time series
#     Ex_min = np.min(Ex_asymptotic)
#     Ex_max = np.max(Ex_asymptotic)
#     In_1_min = np.min(In_1_asymptotic)
#     In_1_max = np.max(In_1_asymptotic)
#     In_2_min = np.min(In_2_asymptotic)  # ADD THIS
#     In_2_max = np.max(In_2_asymptotic)  # ADD THIS

#     return Ex_min, Ex_max, In_1_min, In_1_max, In_2_min, In_2_max  # MODIFY RETURN


# def find_asymptotic_extrema(h_ex_value, model_func, initial_conditions,
#                            t_transient=500, t_analysis=200):
#     """
#     Simulate the system and extract ALL local extrema in asymptotic regime.
#     Returns lists of extrema to capture complex multi-timescale oscillations.
#     """
#     t_total = t_transient + t_analysis
    
#     # Solve ODE
#     sol = solve_ivp(model_func, [0, t_total], initial_conditions, 
#                     args=(h_ex_value,), method='LSODA', dense_output=True,
#                     rtol=1e-8, atol=1e-10)

#     # Evaluate in asymptotic regime with higher resolution
#     t_eval = np.linspace(t_transient, t_total, 1000)  # Increase points to capture fast ripples
#     solution = sol.sol(t_eval)

#     Ex_asymptotic   = solution[0, :]
#     In_1_asymptotic = solution[1, :]
#     In_2_asymptotic = solution[2, :]

#     # Find ALL local maxima and minima using peak detection
#     def get_all_extrema(signal, min_prominence=0.01):
#         """Find all local maxima and minima"""
#         # Find maxima
#         max_indices, _ = find_peaks(signal, prominence=min_prominence)
#         maxima = signal[max_indices] if len(max_indices) > 0 else np.array([np.max(signal)])
        
#         # Find minima (peaks of inverted signal)
#         min_indices, _ = find_peaks(-signal, prominence=min_prominence)
#         minima = signal[min_indices] if len(min_indices) > 0 else np.array([np.min(signal)])
        
#         return minima, maxima
    
#     Ex_mins, Ex_maxs = get_all_extrema(Ex_asymptotic)
#     In_1_mins, In_1_maxs = get_all_extrema(In_1_asymptotic)
#     In_2_mins, In_2_maxs = get_all_extrema(In_2_asymptotic)

#     return Ex_mins, Ex_maxs, In_1_mins, In_1_maxs, In_2_mins, In_2_maxs

def find_asymptotic_extrema(h_ex_value, model_func, initial_conditions,
                           t_transient=500, t_analysis=200):
    """
    Simulate the system and extract ALL local extrema in asymptotic regime.
    Returns separate lists for minima and maxima.
    """
    t_total = t_transient + t_analysis
    
    # Solve ODE
    sol = solve_ivp(model_func, [0, t_total], initial_conditions, 
                    args=(h_ex_value,), method='LSODA', dense_output=True,
                    rtol=1e-8, atol=1e-10)

    # Evaluate in asymptotic regime with higher resolution
    t_eval = np.linspace(t_transient, t_total, 1000)
    solution = sol.sol(t_eval)

    Ex_asymptotic   = solution[0, :]
    In_1_asymptotic = solution[1, :]
    In_2_asymptotic = solution[2, :]

    # Find ALL local maxima and minima using peak detection
    def get_all_extrema(signal, min_prominence=0.01):
        """Find all local maxima and minima - RETURN SEPARATELY"""
        # Find maxima
        max_indices, _ = find_peaks(signal, prominence=min_prominence)
        maxima = signal[max_indices] if len(max_indices) > 0 else np.array([np.max(signal)])
        
        # Find minima (peaks of inverted signal)
        min_indices, _ = find_peaks(-signal, prominence=min_prominence)
        minima = signal[min_indices] if len(min_indices) > 0 else np.array([np.min(signal)])
        
        return minima, maxima  # Return separately!
    
    Ex_mins, Ex_maxs = get_all_extrema(Ex_asymptotic)
    In_1_mins, In_1_maxs = get_all_extrema(In_1_asymptotic)
    In_2_mins, In_2_maxs = get_all_extrema(In_2_asymptotic)

    return Ex_mins, Ex_maxs, In_1_mins, In_1_maxs, In_2_mins, In_2_maxs

def create_bifurcation_diagram(h_ex_range, model_func, initial_conditions, 
                               n_points=100, t_transient=500, t_analysis=200):
    """Create bifurcation diagram data with ALL extrema."""
    h_ex_values = np.linspace(h_ex_range[0], h_ex_range[1], n_points)

    # Store lists of minima and maxima separately
    Ex_mins_list = []
    Ex_maxs_list = []
    In_1_mins_list = []
    In_1_maxs_list = []
    In_2_mins_list = []
    In_2_maxs_list = []

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
            Ex_mins, Ex_maxs, In_1_mins, In_1_maxs, In_2_mins, In_2_maxs = find_asymptotic_extrema(
                h_ex, model_func, current_ic,
                t_transient=t_transient, t_analysis=t_analysis
            )
            
            Ex_mins_list.append(Ex_mins)
            Ex_maxs_list.append(Ex_maxs)
            In_1_mins_list.append(In_1_mins)
            In_1_maxs_list.append(In_1_maxs)
            In_2_mins_list.append(In_2_mins)
            In_2_maxs_list.append(In_2_maxs)
            
            # Update initial conditions for continuation (use means)
            Ex_all = np.concatenate([Ex_mins, Ex_maxs])
            In_1_all = np.concatenate([In_1_mins, In_1_maxs])
            In_2_all = np.concatenate([In_2_mins, In_2_maxs])
            current_ic = [np.mean(Ex_all), np.mean(In_1_all), np.mean(In_2_all)]
            
        except Exception as e:
            print(f"Warning: Failed at h_ex = {h_ex:.6f}: {e}")
            # Use previous values
            if len(Ex_mins_list) > 0:
                Ex_mins_list.append(Ex_mins_list[-1])
                Ex_maxs_list.append(Ex_maxs_list[-1])
                In_1_mins_list.append(In_1_mins_list[-1])
                In_1_maxs_list.append(In_1_maxs_list[-1])
                In_2_mins_list.append(In_2_mins_list[-1])
                In_2_maxs_list.append(In_2_maxs_list[-1])
            else:
                Ex_mins_list.append(np.array([np.nan]))
                Ex_maxs_list.append(np.array([np.nan]))
                In_1_mins_list.append(np.array([np.nan]))
                In_1_maxs_list.append(np.array([np.nan]))
                In_2_mins_list.append(np.array([np.nan]))
                In_2_maxs_list.append(np.array([np.nan]))

    print("-" * 60)
    print("Bifurcation diagram computation complete!")

    return (h_ex_values, Ex_mins_list, Ex_maxs_list, 
            In_1_mins_list, In_1_maxs_list, In_2_mins_list, In_2_maxs_list)

# def create_bifurcation_diagram(h_ex_range, model_func, initial_conditions, 
#                                n_points=100, t_transient=500, t_analysis=200):
#     """Create bifurcation diagram data with ALL extrema."""
#     h_ex_values = np.linspace(h_ex_range[0], h_ex_range[1], n_points)

#     # Store lists of extrema for each parameter value
#     Ex_extrema = []
#     In_1_extrema = []
#     In_2_extrema = []

#     # Use continuation
#     current_ic = initial_conditions.copy()

#     print("Computing bifurcation diagram...")
#     print(f"Scanning h_ex from {h_ex_range[0]:.4f} to {h_ex_range[1]:.4f}")
#     print(f"Number of points: {n_points}")
#     print("-" * 60)

#     for i, h_ex in enumerate(h_ex_values):
#         if (i + 1) % 10 == 0 or i == 0:
#             print(f"Progress: {i+1:3d}/{n_points} | h_ex = {h_ex:.6f}")
        
#         try:
#             Ex_mins, Ex_maxs, In_1_mins, In_1_maxs, In_2_mins, In_2_maxs = find_asymptotic_extrema(
#                 h_ex, model_func, current_ic,
#                 t_transient=t_transient, t_analysis=t_analysis
#             )
            
#             # Store all extrema (both mins and maxs)
#             Ex_all = np.concatenate([Ex_mins, Ex_maxs])
#             In_1_all = np.concatenate([In_1_mins, In_1_maxs])
#             In_2_all = np.concatenate([In_2_mins, In_2_maxs])
            
#             Ex_extrema.append(Ex_all)
#             In_1_extrema.append(In_1_all)
#             In_2_extrema.append(In_2_all)
            
#             # Update initial conditions for continuation (use means)
#             current_ic = [np.mean(Ex_all), np.mean(In_1_all), np.mean(In_2_all)]
            
#         except Exception as e:
#             print(f"Warning: Failed at h_ex = {h_ex:.6f}: {e}")
#             # Use previous values
#             if len(Ex_extrema) > 0:
#                 Ex_extrema.append(Ex_extrema[-1])
#                 In_1_extrema.append(In_1_extrema[-1])
#                 In_2_extrema.append(In_2_extrema[-1])
#             else:
#                 Ex_extrema.append(np.array([np.nan]))
#                 In_1_extrema.append(np.array([np.nan]))
#                 In_2_extrema.append(np.array([np.nan]))

#     print("-" * 60)
#     print("Bifurcation diagram computation complete!")

#     return (h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema)

# def create_bifurcation_diagram(h_ex_range, model_func, initial_conditions, n_points=100, t_transient=500,
#                                t_analysis=200):
#     """Create bifurcation diagram data."""
    
#     h_ex_values = np.linspace(h_ex_range[0], h_ex_range[1], n_points)
    
#     Ex_mins = []
#     Ex_maxs = []
#     In_1_mins = []
#     In_1_maxs = []
#     In_2_mins = []
#     In_2_maxs = []

#     # Use continuation
#     current_ic = initial_conditions.copy()
    
#     print("Computing bifurcation diagram...")
#     print(f"Scanning h_ex from {h_ex_range[0]:.4f} to {h_ex_range[1]:.4f}")
#     print(f"Number of points: {n_points}")
#     print("-" * 60)
    
#     for i, h_ex in enumerate(h_ex_values):
#         if (i + 1) % 10 == 0 or i == 0:
#             print(f"Progress: {i+1:3d}/{n_points} | h_ex = {h_ex:.6f}")
        
#         try:
#             Ex_min, Ex_max, In_1_min, In_1_max, In_2_min, In_2_max = find_asymptotic_extrema(
#                 h_ex, model_func, current_ic,
#                 t_transient=t_transient, t_analysis=t_analysis
#             )
#             # Ex_min, Ex_max, In_1_min, In_1_max = find_asymptotic_extrema(
#             #     h_ex, model_func, current_ic,
#             #     t_transient=t_transient, t_analysis=t_analysis
#             # )
            
#             Ex_mins.append(Ex_min)
#             Ex_maxs.append(Ex_max)
#             In_1_mins.append(In_1_min)
#             In_1_maxs.append(In_1_max)
            
#             # Update initial conditions for continuation
#             # current_ic = [(Ex_min + Ex_max)/2, (In_1_min + In_1_max)/2]
#             current_ic = [(Ex_min + Ex_max)/2, (In_1_min + In_1_max)/2, (In_2_min + In_2_max)/2]

            
#         except Exception as e:
#             print(f"Warning: Failed at h_ex = {h_ex:.6f}: {e}")
#             # Use previous values
#             if len(Ex_mins) > 0:
#                 Ex_mins.append(Ex_mins[-1])
#                 Ex_maxs.append(Ex_maxs[-1])
#                 In_1_mins.append(In_1_mins[-1])
#                 In_1_maxs.append(In_1_maxs[-1])
#             else:
#                 Ex_mins.append(np.nan)
#                 Ex_maxs.append(np.nan)
#                 In_1_mins.append(np.nan)
#                 In_1_maxs.append(np.nan)
    
#     print("-" * 60)
#     print("Bifurcation diagram computation complete!")
    
#     return (np.array(h_ex_values), 
#             np.array(Ex_mins), np.array(Ex_maxs),
#             np.array(In_1_mins), np.array(In_1_maxs))


# def plot_bifurcation_diagram(h_ex_values, Ex_mins, Ex_maxs, In_1_mins, In_1_maxs, 
#                              save_filename='bifurcation_diagram_3V.png',
#                              oscillation_range=None):
#     """
#     Bifurcation diagram with BOTH min and max.
#     """
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
#     # Highlight oscillation region if provided
#     if oscillation_range is not None:
#         ax1.axvspan(oscillation_range[0], oscillation_range[1], 
#                    alpha=0.15, color='yellow', label='Expected oscillations')
#         ax2.axvspan(oscillation_range[0], oscillation_range[1], 
#                    alpha=0.15, color='yellow', label='Expected oscillations')
    
#     # Ex bifurcation diagram
#     # Plot minima in BLUE and maxima in RED
#     ax1.plot(h_ex_values, Ex_mins, 'o', color='blue', markersize=4, 
#              alpha=0.6, label='S min', markeredgewidth=0)
#     ax1.plot(h_ex_values, Ex_maxs, 'o', color='red', markersize=4, 
#              alpha=0.6, label='S max', markeredgewidth=0)
    
#     ax1.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Ex', fontsize=13, fontweight='bold')
#     ax1.set_title('Bifurcation Diagram - Ex', 
#                   fontsize=15, fontweight='bold')
#     ax1.grid(True, alpha=0.3, linestyle='--')
#     ax1.legend(fontsize=11, loc='best')
#     ax1.tick_params(labelsize=11)
    
#     # Plot product bifurcation diagram
#     ax2.plot(h_ex_values, In_1_mins, 'o', color='blue', markersize=4, 
#              alpha=0.6, label='In min', markeredgewidth=0)
#     ax2.plot(h_ex_values, In_1_maxs, 'o', color='red', markersize=4, 
#              alpha=0.6, label='In max', markeredgewidth=0)
    
#     ax2.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('In_1', fontsize=13, fontweight='bold')
#     ax2.set_title('Bifurcation Diagram - In_1', 
#                   fontsize=15, fontweight='bold')
#     ax2.grid(True, alpha=0.3, linestyle='--')
#     ax2.legend(fontsize=11, loc='best')
#     ax2.tick_params(labelsize=11)
    
#     plt.tight_layout()
#     # plt.savefig(save_filename, dpi=300, bbox_inches='tight')
#     # print(f"\nFigure saved as: {save_filename}")
    
#     return fig

# def plot_bifurcation_diagram(h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema,
#                             save_filename='bifurcation_diagram_3V.png'):
#     """
#     Plot bifurcation diagram with ALL extrema shown.
#     """
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
#     # Plot Ex - each parameter value may have multiple extrema
#     for h_ex, extrema in zip(h_ex_values, Ex_extrema):
#         ax1.plot([h_ex]*len(extrema), extrema, 'o', color='darkblue', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax1.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Ex', fontsize=13, fontweight='bold')
#     ax1.set_title('Bifurcation Diagram - Ex (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax1.grid(True, alpha=0.3, linestyle='--')
#     ax1.tick_params(labelsize=11)
    
#     # Plot In_1
#     for h_ex, extrema in zip(h_ex_values, In_1_extrema):
#         ax2.plot([h_ex]*len(extrema), extrema, 'o', color='darkred', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax2.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('In_1', fontsize=13, fontweight='bold')
#     ax2.set_title('Bifurcation Diagram - In_1 (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax2.grid(True, alpha=0.3, linestyle='--')
#     ax2.tick_params(labelsize=11)
    
#     # Plot In_2
#     for h_ex, extrema in zip(h_ex_values, In_2_extrema):
#         ax3.plot([h_ex]*len(extrema), extrema, 'o', color='darkgreen', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax3.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax3.set_ylabel('In_2', fontsize=13, fontweight='bold')
#     ax3.set_title('Bifurcation Diagram - In_2 (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax3.grid(True, alpha=0.3, linestyle='--')
#     ax3.tick_params(labelsize=11)
    
#     plt.tight_layout()
    
#     return fig

def plot_bifurcation_diagram(h_ex_values, Ex_mins_list, Ex_maxs_list, 
                            In_1_mins_list, In_1_maxs_list, In_2_mins_list, In_2_maxs_list,
                            save_filename='bifurcation_diagram_3V.png',
                            oscillation_range=None):
    """
    Plot bifurcation diagram with minima and maxima in different colors.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    
    # Highlight oscillation region if provided
    if oscillation_range is not None:
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(oscillation_range[0], oscillation_range[1], 
                      alpha=0.15, color='yellow', label='Oscillations')
    
    # Plot Ex - minima in blue, maxima in red
    for h_ex, mins, maxs in zip(h_ex_values, Ex_mins_list, Ex_maxs_list):
        ax1.plot([h_ex]*len(mins), mins, 'o', color='blue', 
                markersize=2, alpha=0.6, markeredgewidth=0)
        ax1.plot([h_ex]*len(maxs), maxs, 'o', color='red', 
                markersize=2, alpha=0.6, markeredgewidth=0)
    
    ax1.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Ex', fontsize=13, fontweight='bold')
    ax1.set_title('Bifurcation Diagram - Ex (blue=min, red=max)', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # Plot In_1 - minima in blue, maxima in red
    for h_ex, mins, maxs in zip(h_ex_values, In_1_mins_list, In_1_maxs_list):
        ax2.plot([h_ex]*len(mins), mins, 'o', color='blue', 
                markersize=2, alpha=0.6, markeredgewidth=0)
        ax2.plot([h_ex]*len(maxs), maxs, 'o', color='red', 
                markersize=2, alpha=0.6, markeredgewidth=0)
    
    ax2.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
    ax2.set_ylabel('In_1', fontsize=13, fontweight='bold')
    ax2.set_title('Bifurcation Diagram - In_1 (blue=min, red=max)', 
                  fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)
    
    # Plot In_2 - minima in blue, maxima in red
    for h_ex, mins, maxs in zip(h_ex_values, In_2_mins_list, In_2_maxs_list):
        ax3.plot([h_ex]*len(mins), mins, 'o', color='blue', 
                markersize=2, alpha=0.6, markeredgewidth=0)
        ax3.plot([h_ex]*len(maxs), maxs, 'o', color='red', 
                markersize=2, alpha=0.6, markeredgewidth=0)
    
    ax3.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
    ax3.set_ylabel('In_2', fontsize=13, fontweight='bold')
    ax3.set_title('Bifurcation Diagram - In_2 (blue=min, red=max)', 
                  fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    return fig

# def plot_bifurcation_diagram(h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema,
#                             save_filename='bifurcation_diagram_3V.png',
#                             oscillation_range=None):
#     """
#     Plot bifurcation diagram with ALL extrema shown.
#     """
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    
#     # Highlight oscillation region if provided
#     if oscillation_range is not None:
#         for ax in [ax1, ax2, ax3]:
#             ax.axvspan(oscillation_range[0], oscillation_range[1], 
#                       alpha=0.15, color='yellow', label='Oscillations')
    
#     # Plot Ex - each parameter value may have multiple extrema
#     for h_ex, extrema in zip(h_ex_values, Ex_extrema):
#         ax1.plot([h_ex]*len(extrema), extrema, 'o', color='darkblue', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax1.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Ex', fontsize=13, fontweight='bold')
#     ax1.set_title('Bifurcation Diagram - Ex (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax1.grid(True, alpha=0.3, linestyle='--')
#     ax1.tick_params(labelsize=11)
#     if oscillation_range is not None:
#         ax1.legend(fontsize=11, loc='best')
    
#     # Plot In_1
#     for h_ex, extrema in zip(h_ex_values, In_1_extrema):
#         ax2.plot([h_ex]*len(extrema), extrema, 'o', color='darkred', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax2.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('In_1', fontsize=13, fontweight='bold')
#     ax2.set_title('Bifurcation Diagram - In_1 (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax2.grid(True, alpha=0.3, linestyle='--')
#     ax2.tick_params(labelsize=11)
    
#     # Plot In_2
#     for h_ex, extrema in zip(h_ex_values, In_2_extrema):
#         ax3.plot([h_ex]*len(extrema), extrema, 'o', color='darkgreen', 
#                 markersize=2, alpha=0.6, markeredgewidth=0)
    
#     ax3.set_xlabel('Parameter h_ex', fontsize=13, fontweight='bold')
#     ax3.set_ylabel('In_2', fontsize=13, fontweight='bold')
#     ax3.set_title('Bifurcation Diagram - In_2 (all extrema)', 
#                   fontsize=15, fontweight='bold')
#     ax3.grid(True, alpha=0.3, linestyle='--')
#     ax3.tick_params(labelsize=11)
    
#     plt.tight_layout()
    
#     return fig

def analyze_oscillations(h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema):
    """Analyze where oscillations occur and their complexity."""
    
    # Calculate the number of extrema and amplitude for each parameter value
    Ex_n_extrema = np.array([len(e) for e in Ex_extrema])
    In_1_n_extrema = np.array([len(e) for e in In_1_extrema])
    In_2_n_extrema = np.array([len(e) for e in In_2_extrema])
    
    Ex_amplitudes = np.array([np.max(e) - np.min(e) if len(e) > 0 else 0 for e in Ex_extrema])
    In_1_amplitudes = np.array([np.max(e) - np.min(e) if len(e) > 0 else 0 for e in In_1_extrema])
    In_2_amplitudes = np.array([np.max(e) - np.min(e) if len(e) > 0 else 0 for e in In_2_extrema])
    
    # Threshold for considering it an oscillation
    amplitude_threshold = 0.01
    extrema_threshold = 2  # At least 2 extrema means oscillation
    
    Ex_oscillating = (Ex_amplitudes > amplitude_threshold) & (Ex_n_extrema >= extrema_threshold)
    In_1_oscillating = (In_1_amplitudes > amplitude_threshold) & (In_1_n_extrema >= extrema_threshold)
    In_2_oscillating = (In_2_amplitudes > amplitude_threshold) & (In_2_n_extrema >= extrema_threshold)
    
    print("\n" + "="*70)
    print("OSCILLATION ANALYSIS")
    print("="*70)
    
    # Ex analysis
    if np.any(Ex_oscillating):
        osc_indices = np.where(Ex_oscillating)[0]
        h_ex_osc_start = h_ex_values[osc_indices[0]]
        h_ex_osc_end   = h_ex_values[osc_indices[-1]]
        print(f"\nEx oscillations detected:")
        print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(Ex_amplitudes)
        print(f"  Maximum amplitude: {Ex_amplitudes[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
        
        # Complexity analysis - count extrema in oscillatory region
        max_extrema = np.max(Ex_n_extrema[osc_indices])
        min_extrema = np.min(Ex_n_extrema[osc_indices])
        mean_extrema = np.mean(Ex_n_extrema[osc_indices])
        print(f"  Extrema per cycle: min={min_extrema}, max={max_extrema}, mean={mean_extrema:.1f}")
        
        # Identify regions with complex oscillations (many extrema)
        complex_osc = Ex_n_extrema > 4  # More than 2 max + 2 min suggests fast ripples
        if np.any(complex_osc & Ex_oscillating):
            complex_indices = np.where(complex_osc & Ex_oscillating)[0]
            print(f"  Complex oscillations (>4 extrema): {len(complex_indices)} points")
            print(f"    Range: h_ex ∈ [{h_ex_values[complex_indices[0]]:.4f}, {h_ex_values[complex_indices[-1]]:.4f}]")
    else:
        print("\nNo Ex oscillations detected (all steady state)")
    
    # In_1 analysis
    if np.any(In_1_oscillating):
        osc_indices = np.where(In_1_oscillating)[0]
        h_ex_osc_start = h_ex_values[osc_indices[0]]
        h_ex_osc_end = h_ex_values[osc_indices[-1]]
        print(f"\nIn_1 oscillations detected:")
        print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(In_1_amplitudes)
        print(f"  Maximum amplitude: {In_1_amplitudes[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
        
        # Complexity analysis
        max_extrema = np.max(In_1_n_extrema[osc_indices])
        min_extrema = np.min(In_1_n_extrema[osc_indices])
        mean_extrema = np.mean(In_1_n_extrema[osc_indices])
        print(f"  Extrema per cycle: min={min_extrema}, max={max_extrema}, mean={mean_extrema:.1f}")
        
        # Identify regions with complex oscillations
        complex_osc = In_1_n_extrema > 4
        if np.any(complex_osc & In_1_oscillating):
            complex_indices = np.where(complex_osc & In_1_oscillating)[0]
            print(f"  Complex oscillations (>4 extrema): {len(complex_indices)} points")
            print(f"    Range: h_ex ∈ [{h_ex_values[complex_indices[0]]:.4f}, {h_ex_values[complex_indices[-1]]:.4f}]")
    else:
        print("\nNo In_1 oscillations detected (all steady state)")
    
    # In_2 analysis
    if np.any(In_2_oscillating):
        osc_indices = np.where(In_2_oscillating)[0]
        h_ex_osc_start = h_ex_values[osc_indices[0]]
        h_ex_osc_end = h_ex_values[osc_indices[-1]]
        print(f"\nIn_2 oscillations detected:")
        print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(In_2_amplitudes)
        print(f"  Maximum amplitude: {In_2_amplitudes[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
        
        # Complexity analysis
        max_extrema = np.max(In_2_n_extrema[osc_indices])
        min_extrema = np.min(In_2_n_extrema[osc_indices])
        mean_extrema = np.mean(In_2_n_extrema[osc_indices])
        print(f"  Extrema per cycle: min={min_extrema}, max={max_extrema}, mean={mean_extrema:.1f}")
        
        # Identify regions with complex oscillations
        complex_osc = In_2_n_extrema > 4
        if np.any(complex_osc & In_2_oscillating):
            complex_indices = np.where(complex_osc & In_2_oscillating)[0]
            print(f"  Complex oscillations (>4 extrema): {len(complex_indices)} points")
            print(f"    Range: h_ex ∈ [{h_ex_values[complex_indices[0]]:.4f}, {h_ex_values[complex_indices[-1]]:.4f}]")
    else:
        print("\nNo In_2 oscillations detected (all steady state)")
    
    print("="*70 + "\n")
    
    return {
        'Ex': {'oscillating': Ex_oscillating, 'amplitudes': Ex_amplitudes, 'n_extrema': Ex_n_extrema},
        'In_1': {'oscillating': In_1_oscillating, 'amplitudes': In_1_amplitudes, 'n_extrema': In_1_n_extrema},
        'In_2': {'oscillating': In_2_oscillating, 'amplitudes': In_2_amplitudes, 'n_extrema': In_2_n_extrema}
    }

# def analyze_oscillations(h_ex_values, Ex_mins, Ex_maxs, In_1_mins, In_1_maxs):
#     """Analyze where oscillations occur."""
    
#     Ex_amplitude = Ex_maxs - Ex_mins
#     In_1_amplitude = In_1_maxs - In_1_mins
    
#     # Threshold for considering it an oscillation
#     threshold = 0.01
    
#     Ex_oscillating = Ex_amplitude > threshold
#     In_1_oscillating = In_1_amplitude > threshold
    
#     print("\n" + "="*70)
#     print("OSCILLATION ANALYSIS")
#     print("="*70)
    
#     # Find oscillatory regions
#     if np.any(Ex_oscillating):
#         osc_indices = np.where(Ex_oscillating)[0]
#         h_ex_osc_start = h_ex_values[osc_indices[0]]
#         h_ex_osc_end   = h_ex_values[osc_indices[-1]]
#         print(f"\Ex oscillations detected:")
#         print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
#         print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
#         # Max amplitude
#         max_amp_idx = np.argmax(Ex_amplitude)
#         print(f"  Maximum amplitude: {Ex_amplitude[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
#     else:
#         print("\nNo Ex oscillations detected (all steady state)")
    
#     if np.any(In_1_oscillating):
#         osc_indices = np.where(In_1_oscillating)[0]
#         h_ex_osc_start = h_ex_values[osc_indices[0]]
#         h_ex_osc_end = h_ex_values[osc_indices[-1]]
#         print(f"\nProduct oscillations detected:")
#         print(f"  Range: h_ex ∈ [{h_ex_osc_start:.4f}, {h_ex_osc_end:.4f}]")
#         print(f"  Number of oscillating points: {len(osc_indices)}/{len(h_ex_values)}")
        
#         # Max amplitude
#         max_amp_idx = np.argmax(In_1_amplitude)
#         print(f"  Maximum amplitude: {In_1_amplitude[max_amp_idx]:.4f} at h_ex = {h_ex_values[max_amp_idx]:.4f}")
#     else:
#         print("\nNo product oscillations detected (all steady state)")
    
#     print("="*70 + "\n")




# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("BIFURCATION ANALYSIS")
    print("E-I1-I2 Neural Population Model")
    print("="*70)
    
    # Model parameters
    Ex_0 = 1
    In_1_0 = -1
    In_2_0 = -0 
    initial_conditions = [Ex_0, In_1_0, In_2_0]
    
    # Bifurcation parameter range
    h_ex_min = -3
    h_ex_max = -0
    n_points = 150
    
    # Simulation parameters
    t_transient = 80
    t_analysis  = 80
    
    print(f"\nParameters:")
    print(f"  h_ex range: [{h_ex_min}, {h_ex_max}]")
    print(f"  Points: {n_points}")
    print(f"  Initial conditions: Ex_0={Ex_0}, In_1_0={In_1_0}, In_2_0={In_2_0}")
    print()
    
    # Create bifurcation diagram
    h_ex_values, Ex_mins_list, Ex_maxs_list, In_1_mins_list, In_1_maxs_list, In_2_mins_list, In_2_maxs_list = create_bifurcation_diagram(
        [h_ex_min, h_ex_max],
        dX_dt,
        initial_conditions,
        n_points=n_points,
        t_transient=t_transient,
        t_analysis=t_analysis
    )
    
    # Plot
    fig = plot_bifurcation_diagram(h_ex_values, Ex_mins_list, Ex_maxs_list, In_1_mins_list, In_1_maxs_list, In_2_mins_list, In_2_maxs_list,
                                   save_filename='EI_bifurcation_diagram.png',
                                   oscillation_range=[-2.45, -0.3])
    
    fig.savefig('Bifurcations_Bursting.png', format='png')

    # Analyze
    # analyze_oscillations(h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema)
    
    # # Print some sample values
    # print("\nSample values:")
    # print("h_ex      | Ex_min  | Ex_max  | In_1_min  | In_1_max  ")
    # print("-" * 55)
    # # for i in [0, 10, 50, 100, 149]:
    # #     print(f"{h_ex_values[i]:.4f} | {Ex_mins[i]:.4f} | {Ex_maxs[i]:.4f} | {In_1_mins[i]:.4f} | {In_1_maxs[i]:.4f}")

# For analyze_oscillations, combine mins and maxs
Ex_extrema = [np.concatenate([mins, maxs]) for mins, maxs in zip(Ex_mins_list, Ex_maxs_list)]
In_1_extrema = [np.concatenate([mins, maxs]) for mins, maxs in zip(In_1_mins_list, In_1_maxs_list)]
In_2_extrema = [np.concatenate([mins, maxs]) for mins, maxs in zip(In_2_mins_list, In_2_maxs_list)]

# Analyze
oscillation_info = analyze_oscillations(h_ex_values, Ex_extrema, In_1_extrema, In_2_extrema)

# Print some sample values showing complexity
print("\nSample values (showing complexity):")
print("h_ex      | Ex extrema | In_1 extrema | In_2 extrema | Ex amp  ")
print("-" * 70)
for i in [0, 10, 50, 100, 149]:
    if i < len(h_ex_values):
        print(f"{h_ex_values[i]:.4f} | {len(Ex_extrema[i]):^10} | {len(In_1_extrema[i]):^12} | "
              f"{len(In_2_extrema[i]):^12} | {oscillation_info['Ex']['amplitudes'][i]:.4f}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    plt.show()
