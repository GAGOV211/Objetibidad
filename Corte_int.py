import numpy as np
import matplotlib.pyplot as plt

# Definir una lista de valores de theta
L=1
E=100
nu=0.3
Nmesh=10
Aprox="Finite"
Modelo="Hyper"
D_max=15
D_values = np.linspace(0, L*0.46, D_max)

# Listas para almacenar los resultados
integral_energia_deformacion = 0
integral_strain_efectivo = 0
integral_stress_efectivo = 0

integral_energia_deformacion = []
integral_strain_efectivo = []
integral_stress_efectivo = []

# Calcular las integrales para cada valor de theta
for D in D_values:
    # Resolver el problema elástico y calcular las cantidades necesarias
    desplazamiento = resolver_problema_elastico_C(E, nu,D, Nmesh, L, Aprox, Modelo)
    energia_deformacion = calcular_energia_desformacion(desplazamiento, E, nu,  Aprox, Modelo)
    strain_efectivo = calcular_deformacion_efectiva(desplazamiento, E, nu,  Aprox, Modelo)
    stress_efectivo = calcular_tension_efectiva(desplazamiento, E, nu,  Aprox, Modelo)
    
    # Calcular las integrales sobre el dominio
    integral_energia_deformacion.append(fe.assemble(energia_deformacion * fe.dx))
    integral_strain_efectivo.append(fe.assemble(strain_efectivo * fe.dx))
    integral_stress_efectivo.append(fe.assemble(stress_efectivo * fe.dx))

# Convertir las listas en arrays de NumPy
integral_energia_deformacion = np.array(integral_energia_deformacion)
integral_strain_efectivo = np.array(integral_strain_efectivo)
integral_stress_efectivo = np.array(integral_stress_efectivo)

# Graficar los resultados
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la integral de energía de deformación
ax1.plot(D_values, integral_energia_deformacion, '--', color='black', label='Energía de Deformación')
ax1.set_xlabel('Desplazamiento')
ax1.set_ylabel('Energía de Deformación', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Crear un segundo eje y comparte el mismo x
ax2 = ax1.twinx()

# Graficar el stress efectivo
ax2.scatter(D_values, integral_stress_efectivo, marker='o', facecolors='none', edgecolors='tab:red', s=100, label='Stress Efectivo')
ax2.set_ylabel('Stress Efectivo', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Crear un tercer eje y comparte el mismo x
ax3 = ax1.twinx()

# Graficar el strain efectivo
ax3.scatter(D_values, integral_strain_efectivo, marker='x', color='tab:green', s=100, label='Strain Efectivo')
ax3.set_ylabel('Strain Efectivo', color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')

# Ajustar la posición del tercer eje
ax3.spines['right'].set_position(('outward', 50))

# Ajustar el diseño de la leyenda
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines = lines1 + lines2 + lines3
labels = labels1 + labels2 + labels3
ax1.legend(lines, labels, loc='lower right')

fig.tight_layout()
plt.show()

fe.plot(desplazamiento,mode="displacement")
