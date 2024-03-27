
# Crear una lista para almacenar los resultados proyectados
resultados = []

# Definir una lista de valores de theta
L = 1
E = 100
nu = 0.40
Nmesh = 20
Aprox = "Finite"
Modelo = "Hyper"

# Definir la cantidad máxima de desplazamiento
desplazamiento_maximo = L * 0.34

# Resolver el problema elástico para diferentes valores de desplazamiento
for desplazamiento in np.linspace(0, desplazamiento_maximo, 20):
    # Resolver el problema elástico para obtener los desplazamientos
    u_solution = resolver_problema_elastico_T(E, nu, desplazamiento, Nmesh, L, Aprox, Modelo)
    
    # Crear el espacio de funciones para proyectar
    V = fe.FunctionSpace(u_solution.function_space().mesh(), "CG", 1)
    
    # Crear una función de prueba y un punto donde se proyectarán las funciones
    v = fe.TestFunction(V)
    point = fe.Point(0.5, 0.5)  # Punto en el dominio donde se proyectarán las funciones
    
    # Proyectar la función de esfuerzo en el punto
    stress_projection = fe.project(calcular_tension_efectiva(u_solution, E, nu, Aprox, Modelo), V)
    stress_at_point = stress_projection(point)
    
    # Proyectar la función de deformación en el punto
    strain_projection = fe.project(calcular_deformacion_efectiva(u_solution, E, nu, Aprox, Modelo), V)
    strain_at_point = strain_projection(point)
    
    # Proyectar el stress en YY en el punto
    stress_YY_projection = fe.project(calcular_tension_YY(u_solution, E, nu, Aprox, Modelo), V)
    stress_YY_at_point = stress_YY_projection(point)
    
    # Proyectar el strain en YY en el punto
    strain_YY_projection = fe.project(calcular_deformacion_YY(u_solution, E, nu, Aprox), V)
    strain_YY_at_point = strain_YY_projection(point)
    
    # Agregar los valores proyectados a la lista de resultados
    resultados.append((desplazamiento / L, strain_at_point, stress_at_point, strain_YY_at_point, stress_YY_at_point))

# Extraer los resultados de la lista de resultados
desplazamientos_L = [resultado[0] for resultado in resultados]
strains_efectivo = [resultado[1] for resultado in resultados]
stresses_efectivo = [resultado[2] for resultado in resultados]
strains_YY = [resultado[3] for resultado in resultados]
stresses_YY = [resultado[4] for resultado in resultados]

# Graficar los resultados en dos filas y tres columnas
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

# Graficar stress efectivo versus strain efectivo
ax[0, 0].plot(strains_efectivo, stresses_efectivo, marker='s', linestyle='--', color='r')
ax[0, 0].set_xlabel('Strain efectivo')
ax[0, 0].set_ylabel('Stress efectivo')
ax[0, 0].set_title('Stress efectivo vs. Strain efectivo')
#ax[0, 0].grid(True)

# Graficar stress efectivo versus desplazamiento/L
ax[0, 1].plot(desplazamientos_L, stresses_efectivo, marker='o', linestyle='--', color='g')
ax[0, 1].set_xlabel('Desplazamiento/L')
ax[0, 1].set_ylabel('Stress efectivo')
ax[0, 1].set_title('Stress efectivo vs. Desplazamiento/L')
#ax[0, 1].grid(True)

# Graficar strain efectivo versus desplazamiento/L
ax[0, 2].plot(desplazamientos_L, strains_efectivo, marker='x', linestyle='--', color='b')
ax[0, 2].set_xlabel('Desplazamiento/L')
ax[0, 2].set_ylabel('Strain efectivo')
ax[0, 2].set_title('Strain efectivo vs. Desplazamiento/L')
#ax[0, 2].grid(True)

# Graficar stress YY versus strain YY
ax[1, 0].plot(strains_YY, stresses_YY, marker='s', linestyle='-.', color='Orange')
ax[1, 0].set_xlabel('Strain YY')
ax[1, 0].set_ylabel('Stress YY')
ax[1, 0].set_title('Stress YY vs. Strain YY')
#ax[1, 0].grid(True)

# Graficar stress YY efectivo versus desplazamiento/L
ax[1, 1].plot(desplazamientos_L, stresses_YY, marker='o', linestyle='-.', color='y')
ax[1, 1].set_xlabel('Desplazamiento/L')
ax[1, 1].set_ylabel('Stress YY')
ax[1, 1].set_title('Stress YY vs. Desplazamiento/L')
#ax[1, 1].grid(True)

# Graficar strain YY versus desplazamiento/L
ax[1, 2].plot(desplazamientos_L, strains_YY,  marker='x', linestyle='-.', color='purple')
ax[1, 2].set_xlabel('Desplazamiento/L')
ax[1, 2].set_ylabel('Strain YY')
ax[1, 2].set_title('Strain YY vs. Desplazamiento/L')
#ax[1, 2].grid(True)

# Agregar título a toda la imagen
fig.suptitle('Comparación de Tensiones y Deformaciones en diferentes direcciones\n'+"Modelo:  "+Aprox+" "+Modelo+"Elastico"+"\n")

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar la figura
plt.show()

fe.plot(u_solution, mode="displacement")
