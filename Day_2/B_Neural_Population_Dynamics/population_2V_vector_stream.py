import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from numpy import array, tanh

def dX_dt(X):
    """Return the rates at all positions."""
    h_ex, h_in, tau_ex, tau_in, c2, c4, c_EE, c_EI = (-7.1, -4., 1, 1.5, 10, 0, 5, 10)
    
    return array([
        (h_ex - X[0] - c2*tanh(X[1]) + c_EE*tanh(X[0]))*tau_ex,
        (h_in - X[1] - c4*tanh(X[1]) + c_EI*tanh(X[0]))*tau_in
    ])

# Grid
x = np.linspace(-10, 10, 50)
y = np.linspace(-20, 10, 50)
X, Y = np.meshgrid(x, y)

# Vector field and magnitudes
DX = np.zeros_like(X)
DY = np.zeros_like(Y)
magnitudes = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dX = dX_dt([X[i,j], Y[i,j]])
        DX[i,j] = dX[0]
        DY[i,j] = dX[1]
        magnitudes[i,j] = np.sqrt(dX[0]**2 + dX[1]**2)

# Normalize vectors for integration
speed = np.sqrt(DX**2 + DY**2)
speed[speed == 0] = 1  # Avoid division by zero
DX_norm = DX / speed
DY_norm = DY / speed

# Generate streamlines manually
def generate_streamlines(X, Y, DX_norm, DY_norm, magnitudes, num_streams=150, max_length=3.0):
    """Generate streamline segments by integrating the vector field."""
    np.random.seed(42)
    streamlines = []
    
    # Start points distributed across the domain
    x_starts = np.random.uniform(X.min(), X.max(), num_streams)
    y_starts = np.random.uniform(Y.min(), Y.max(), num_streams)
    
    for x_start, y_start in zip(x_starts, y_starts):
        # Integrate forward to create streamline
        points = [[x_start, y_start]]
        colors = []
        
        x_curr, y_curr = x_start, y_start
        dt = 0.15  # Integration step
        
        for _ in range(int(max_length / dt)):
            # Interpolate vector field at current position
            if x_curr < X.min() or x_curr > X.max() or y_curr < Y.min() or y_curr > Y.max():
                break
            
            # Bilinear interpolation
            i = np.searchsorted(x, x_curr) - 1
            j = np.searchsorted(y, y_curr) - 1
            
            if i < 0 or i >= len(x)-1 or j < 0 or j >= len(y)-1:
                break
            
            # Interpolation weights
            wx = (x_curr - x[i]) / (x[i+1] - x[i])
            wy = (y_curr - y[j]) / (y[j+1] - y[j])
            
            # Interpolate velocity
            dx = (1-wx)*(1-wy)*DX_norm[j,i] + wx*(1-wy)*DX_norm[j,i+1] + \
                 (1-wx)*wy*DX_norm[j+1,i] + wx*wy*DX_norm[j+1,i+1]
            dy = (1-wx)*(1-wy)*DY_norm[j,i] + wx*(1-wy)*DY_norm[j,i+1] + \
                 (1-wx)*wy*DY_norm[j+1,i] + wx*wy*DY_norm[j+1,i+1]
            
            # Interpolate magnitude for coloring
            mag = (1-wx)*(1-wy)*magnitudes[j,i] + wx*(1-wy)*magnitudes[j,i+1] + \
                  (1-wx)*wy*magnitudes[j+1,i] + wx*wy*magnitudes[j+1,i+1]
            
            # Step forward
            x_curr += dx * dt
            y_curr += dy * dt
            points.append([x_curr, y_curr])
            colors.append(mag)
        
        if len(points) > 2:  # Only keep streamlines with enough points
            streamlines.append({
                'points': np.array(points),
                'colors': np.array(colors)
            })
    
    return streamlines

# Animation parameters
NUM_STREAMS = 500  # Adjust this to control resolution (number of trajectories)
NUM_FRAMES = 60    # Number of frames in one cycle
MAX_STREAM_LENGTH = 3.5  # Maximum length of each streamline

# Generate streamlines
print("Generating streamlines...")
streamlines = generate_streamlines(X, Y, DX_norm, DY_norm, magnitudes, 
                                   num_streams=NUM_STREAMS, max_length=MAX_STREAM_LENGTH)
print(f"Generated {len(streamlines)} streamlines")

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 8))
plt.subplots_adjust(bottom=0.25, left=0.1)  # More room for sliders

# Animation state
animation_running = True
current_frame = [0]  # Use list to make it mutable in nested function

# Store original axis limits
x_min_orig, x_max_orig = -10, 10
y_min_orig, y_max_orig = -20, 10

# Create line collections for each streamline
lines = []
arrows = []  # Store arrow patches for when paused
for stream in streamlines:
    line, = ax.plot([], [], linewidth=2, alpha=0.6)
    lines.append(line)
    # Create arrow patch (initially invisible)
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                       visible=False)
    arrows.append(arrow)

# Setup axes
ax.set_xlabel('X₀', fontsize=14)
ax.set_ylabel('X₁', fontsize=14)
ax.set_title('Animated Vector Field - Stream Flux Visualization\n(Dark blue = slow, Yellow = fast)', 
             fontsize=16)
ax.set_xlim([x_min_orig, x_max_orig])
ax.set_ylim([y_min_orig, y_max_orig])
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Colormap
cmap = plt.cm.plasma
vmin = magnitudes.min()
vmax = magnitudes.max()

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Speed')

def update_frame():
    """Update the current frame."""
    # Calculate progress (0 to 1)
    progress = current_frame[0] / NUM_FRAMES
    
    # Update each streamline
    for line, arrow, stream in zip(lines, arrows, streamlines):
        points = stream['points']
        colors = stream['colors']
        
        # Calculate how many points to show based on progress
        num_points = max(2, int(len(points) * progress))
        
        # Get subset of points
        x_data = points[:num_points, 0]
        y_data = points[:num_points, 1]
        
        line.set_data(x_data, y_data)
        
        # Set color based on average magnitude of visible segment
        if len(colors) > 0:
            avg_color = np.mean(colors[:num_points-1]) if num_points > 1 else colors[0]
            normalized_color = (avg_color - vmin) / (vmax - vmin)
            line_color = cmap(normalized_color)
            line.set_color(line_color)
            
            # Update arrow when paused
            if not animation_running and num_points >= 2:
                # Arrow at the tip of the streamline
                tip = points[num_points-1]
                # Previous point for direction
                prev = points[num_points-2]
                arrow.set_visible(True)
                arrow.xy = tip
                arrow.xyann = prev
                arrow.arrowprops['color'] = line_color
            else:
                arrow.set_visible(False)
        else:
            arrow.set_visible(False)
    
    # Advance frame only if animation is running
    if animation_running:
        current_frame[0] = (current_frame[0] + 1) % NUM_FRAMES
    
    fig.canvas.draw_idle()

# Initialize with first frame
update_frame()

# Create timer
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(update_frame)
timer.start()

# Add play/pause button
ax_button = plt.axes([0.45, 0.02, 0.1, 0.04])
button = Button(ax_button, 'Pause')

def toggle_animation(event):
    global animation_running
    animation_running = not animation_running
    button.label.set_text('Play' if not animation_running else 'Pause')
    # Trigger immediate update to show/hide arrows
    update_frame()

button.on_clicked(toggle_animation)

# Add sliders for axis limits
ax_xmin_slider = plt.axes([0.1, 0.14, 0.35, 0.02])
ax_xmax_slider = plt.axes([0.55, 0.14, 0.35, 0.02])
ax_ymin_slider = plt.axes([0.1, 0.10, 0.35, 0.02])
ax_ymax_slider = plt.axes([0.55, 0.10, 0.35, 0.02])

slider_xmin = Slider(ax_xmin_slider, 'X min', -10, 10, valinit=x_min_orig, valstep=0.5)
slider_xmax = Slider(ax_xmax_slider, 'X max', -10, 10, valinit=x_max_orig, valstep=0.5)
slider_ymin = Slider(ax_ymin_slider, 'Y min', -15, 10, valinit=y_min_orig, valstep=0.5)
slider_ymax = Slider(ax_ymax_slider, 'Y max', -15, 10, valinit=y_max_orig, valstep=0.5)

def update_xlim(val):
    xmin = slider_xmin.val
    xmax = slider_xmax.val
    if xmin < xmax:
        ax.set_xlim([xmin, xmax])
        fig.canvas.draw_idle()

def update_ylim(val):
    ymin = slider_ymin.val
    ymax = slider_ymax.val
    if ymin < ymax:
        ax.set_ylim([ymin, ymax])
        fig.canvas.draw_idle()

slider_xmin.on_changed(update_xlim)
slider_xmax.on_changed(update_xlim)
slider_ymin.on_changed(update_ylim)
slider_ymax.on_changed(update_ylim)

plt.show()
