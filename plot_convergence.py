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

plt.figure(figsize=(8,6))
plt.loglog(h, e_u, 'o-', linewidth=2, label='Velocity L2 Error')
plt.loglog(h, e_p, 's-', linewidth=2, label='Pressure L2 Error')
plt.loglog(h, e_c, '^-', linewidth=2, label='Concentration L2 Error')

# Reference slopes derived from finest test mesh
h_slope = np.array([h[-1], h[0]])

# Ideally Taylor-Hood O(h^3) for velocity, O(h^2) for pressure, O(h^3) for quadratic concentration
plt.loglog(h_slope, e_u[-1] * (h_slope/h[-1])**2, 'k--', alpha=0.7, label='O(h^2)')
plt.loglog(h_slope, e_u[-1] * (h_slope/h[-1])**3, 'k:', alpha=0.7, label='O(h^3)')

plt.xlabel('Mesh size (h)', fontsize=12)
plt.ylabel('Discrete L2 Error vs Reference Solution', fontsize=12)
plt.title('Convergence Analysis for 3D Boussinesq flow', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('results/convergence_plot.png', dpi=300)
print("Saved purely static plot to results/convergence_plot.png")
