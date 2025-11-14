import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class Simple2V:
    """
   Linear model with two variables X and Y
    """
    
    def __init__(self, L=32, dt=0.05, dx=1.0, D_x=0.0, D_y=0.0):
        # Grid parameters
        self.L = L
        self.dt = dt
        self.dx = dx
        
        # Diffusion coefficients
        self.D_x = D_x  # X diffusion
        self.D_y = D_y  # Y diffusion
        
        # Model parameters
        self.a = 0.3
        self.b = 0.1
        self.k1 = 0.2
        self.k2 = 0.2

        # Unified stimulation parameters
        self.stim_center = (L//2, L//2) # Center of stimulation
        self.stim_radius = 5           # Radius of stimulation area
        self.offset = 1.0              # Offset for symmetric stimulation
        self.time   = 0.0                # Internal time counter
        
        # Initialize fields
        self.X = np.ones((L, L)) * 0.4  # X
        self.Y = np.ones((L, L)) * 2.7  # Y
        
        # Laplacian kernel for diffusion
        self.laplacian_kernel = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]) / (dx**2)
        
        # Phase map (calculated when phase_shift is set)
        self.phase_map = None
    
    def set_stimulation(self, offset=0.0, center=None):
        """
        Apply constant offset stimulation to the center region
        """
        self.offset = offset
        
        if center is not None:
            self.stim_center = center
            
        self.time = 0.0
    
    def apply_stimulation(self):
        """Apply the appropriate stimulation based on parameters"""
        if self.offset <= 0:
            return
            
        center_x, center_y = self.stim_center
        y, x      = np.ogrid[-center_y:self.L-center_y, -center_x:self.L-center_x]
        distance  = np.sqrt(x*x + y*y)
        stim_mask = distance <= self.stim_radius
        
        # Create stimulation array of the same shape as the grid
        stimulation = np.zeros((self.L, self.L))
        
        # Symmetric stimulation for target patterns
        stimulation[stim_mask] = self.offset
        
        # Apply to X field
        self.X[stim_mask] += stimulation[stim_mask] * self.dt
        self.time += self.dt
    
    def reaction_terms(self, X, Y):
        """Calculate reaction terms"""
        dXdt = (self.a - self.k1*X)
        dYdt = (self.b - self.k2*Y)
    
        return dXdt, dYdt
    
    def diffusion_step(self, field, D):
        """Apply diffusion using convolution"""
        return D * ndimage.convolve(field, self.laplacian_kernel, mode='reflect')
    
    def step(self):
        """Perform one integration step"""
        # Calculate reaction terms
        dX_react, dY_react = self.reaction_terms(self.X, self.Y)
        
        # Add diffusion
        dX_diff = self.diffusion_step(self.X, self.D_x)
        dY_diff = self.diffusion_step(self.Y, self.D_y)
        
        # Update concentrations
        self.X += self.dt * (dX_react + dX_diff)
        self.Y += self.dt * (dY_react + dY_diff)
        
        # Apply stimulation
        self.apply_stimulation()
        
        # Ensure non-negative concentrations
        self.X = np.maximum(self.X, 0)
        self.Y = np.maximum(self.Y, 0)
    
    def add_initial_noise(self, x_noise=0.1, y_noise=0.05):
        """Add initial noise to break symmetry"""
        self.X += np.random.normal(0, x_noise, (self.L, self.L))
        self.Y += np.random.normal(0, y_noise, (self.L, self.L))


def run_simulation(stim_type="target", offset=2, L=32, D_x=1, D_y=0.0, time_steps=5000):
    """
    Unified simulation runner
    
    Parameters:
    - stim_type: "target" or "other" 
    - offset: constant offset
    """
    
    # Create model
    model = Simple2V(L=L, D_x=D_x, D_y=D_y)
    model.add_initial_noise()
    
    # Set stimulation
    if stim_type == "target":
        model.set_stimulation(
            offset=2.0
        )
    elif stim_type == "other":
        model.set_stimulation(
            offset=0
        )
    else:
        raise ValueError("stim_type must be 'target' or 'other'")
    
    # Set up visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fixed color ranges
    X_min, X_max = 0, 10.0
    Y_min, Y_max = 2, 3.0
    
    # Create plots
    im1 = ax1.imshow(model.X, cmap='gray', origin='lower', vmin=X_min, vmax=X_max, interpolation='Gaussian')
    ax1.set_title('X')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(model.Y, cmap='viridis', origin='lower', vmin=Y_min, vmax=Y_max)
    ax2.set_title('Y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Run simulation
    for step in range(time_steps):
        model.step()
        
        if step % 5 == 0:
            im1.set_data(model.X)
            im2.set_data(model.Y)
            ax1.set_title(f'X - Step {step}')
            ax2.set_title(f'Y - Step {step}')
            plt.pause(0.002)
            
        if step % 500 == 0:
            print(f"Step {step}, X range: [{model.X.min():.2f}, {model.X.max():.2f}]")
    
    plt.show()
    return model

# Example usage
if __name__ == "__main__":
    print("2V Linear Model - Central Source")
    
    # Target pattern (symmetric elevated centre)
    model = run_simulation(stim_type="target", offset=2)
