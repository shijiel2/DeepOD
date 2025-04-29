import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)  # You can use any integer value

dataset_path = 'dataset/DCdetector_dataset/UCR/UCR_1_test.npy'
df = pd.DataFrame(np.load(dataset_path, allow_pickle=True))
df_min = df[0].min()
df_max = df[0].max()
df = pd.DataFrame((df[0] - df_min) / (df_max - df_min))

df = df[0:200]
linewidth = 5

# Get y-axis limits to use consistently across all plots
y_min = df[0].min() - 0.05  # Add some padding
y_max = df[0].max() + 0.05  # Add some padding

# Create figure with white background for original data
plt.figure(figsize=(10, 5), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot the time series data
plt.plot(df.index, df[0], linestyle='-', linewidth=linewidth, color='red')

# Set consistent y-axis limits
plt.ylim(y_min, y_max)

# Add grid
plt.grid(False)

# Remove tick labels while keeping the axes and grid
ax.set_xticklabels([])  # Remove x-axis tick labels
ax.set_yticklabels([])  # Remove y-axis tick labels

# Hide axis lines but keep the plot boundaries
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Save the plot with white background
plt.tight_layout()
plt.savefig('time_series_plot.png', dpi=300, bbox_inches='tight')

# Create random noise time series with the same length as df but with pattern
noise_length = len(df)
noise_data = np.zeros(noise_length)
current_idx = 0

while current_idx < noise_length:
    # Decide if this segment should be zeros or noise
    is_zero_segment = np.random.choice([True, False])
    
    # Determine segment length (8-10 timesteps)
    segment_length = np.random.randint(8, 11)
    
    # Make sure we don't exceed the array length
    end_idx = min(current_idx + segment_length, noise_length)
    
    if not is_zero_segment:
        # Fill with random noise between -0.01 and 0.01 (much smaller than the main data scale)
        # This keeps noise visible but maintains the same y-axis scale
        noise_data[current_idx:end_idx] = np.random.uniform(low=-0.01, high=0.01, size=(end_idx - current_idx))
    
    # Move to next segment
    current_idx = end_idx

noise_df = pd.DataFrame(noise_data)

# Create figure with white background for noise plot
plt.figure(figsize=(10, 5), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot the noise data
plt.plot(noise_df.index, noise_df[0], linestyle='-', linewidth=linewidth, color='orange')

# Set consistent y-axis limits
plt.ylim(-0.05, 0.05)

# Add grid
plt.grid(True)

# Remove tick labels while keeping the axes and grid
ax.set_xticklabels([])  # Remove x-axis tick labels
ax.set_yticklabels([])  # Remove y-axis tick labels

# Hide axis lines but keep the plot boundaries
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])

# Save the noise plot with white background
plt.tight_layout()
plt.savefig('noise_plot.png', dpi=300, bbox_inches='tight')

# Add the original time series and noise data together
combined_data = df[0].values + noise_df[0].values
combined_df = pd.DataFrame(combined_data)

# Create figure with white background for combined plot
plt.figure(figsize=(10, 5), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot the combined data
plt.plot(df.index, combined_df[0], linestyle='-', linewidth=linewidth, color='blue')

# Set consistent y-axis limits
plt.ylim(y_min, y_max)

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Remove tick labels while keeping the axes and grid
ax.set_xticklabels([])  # Remove x-axis tick labels
ax.set_yticklabels([])  # Remove y-axis tick labels

# Hide axis lines but keep the plot boundaries
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Save the combined plot with white background
plt.tight_layout()
plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
