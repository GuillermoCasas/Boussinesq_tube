import json
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = 'results/convergence_data.json'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

with open(data_path, 'r') as f:
    data = json.load(f)

h = np.array(data['h_vals'])
e_u = np.array(data['errors_u'])
e_p = np.array(data['errors_p'])
e_c = np.array(data['errors_c'])

# Reference slopes derived from finest test mesh
h_slope = np.array([h[-1], h[0]])
slope_h2 = (h_slope/h[-1])**2
slope_h3 = (h_slope/h[-1])**3

def create_plot(filename, title, y_label, e_data, label, show_h2=True, show_h3=True):
    plt.figure(figsize=(8,6))
    plt.loglog(h, e_data, 'o-', linewidth=2, label=label)
    
    if show_h2:
        plt.loglog(h_slope, e_data[-1] * slope_h2, 'k--', alpha=0.7, label='O(h^2)')
    if show_h3:
        plt.loglog(h_slope, e_data[-1] * slope_h3, 'k:', alpha=0.7, label='O(h^3)')

    plt.xlabel('Mesh size (h)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close()
    print(f"Saved {filename}")

# Velocity Plot
create_plot('convergence_velocity.png', 'Velocity Convergence (3D Boussinesq)', 
            'Velocity L2 Error', e_u, 'Velocity L2', show_h2=False, show_h3=True)

# Pressure Plot
create_plot('convergence_pressure.png', 'Pressure Convergence (3D Boussinesq)', 
            'Pressure L2 Error', e_p, 'Pressure L2', show_h2=True, show_h3=False)

# Concentration Plot
create_plot('convergence_concentration.png', 'Concentration Convergence (3D Boussinesq)', 
            'Concentration L2 Error', e_c, 'Concentration L2', show_h2=False, show_h3=True)

print("Finished generating separate convergence plots.")
