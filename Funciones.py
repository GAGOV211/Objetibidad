
def calcular_epsilon(u):
    
    import numpy as np
    import ufl

    # Tensor de identidad
    I = ufl.Identity(2)
    # Tensor de deformación
    F = I + ufl.grad(u)
    # Tensor de deformación derecho o tensor de Cauchy-Green
    C = ufl.variable(F.T * F)
    
    # Tolerancia
    #tol = DOLFIN_EPS
    tol=0.000000001
    # Determinar perturbación a partir de la tolerancia
    pert = 2 * tol

    # Obtener invariantes requeridos
    I1 = ufl.tr(C)  # traza del tensor
    I3 = ufl.det(C)  # determinante del tensor

    # Valor preliminar para el discriminante
    D = I1**2 - 4 * I3
    # Agregar perturbación numérica al discriminante si está cerca de cero; asegurar positividad de D
    D = ufl.conditional(ufl.lt(D, tol), abs(D) + pert, D)

    lambda1 = 0.5 * (I1 + ufl.sqrt(D))
    lambda2 = I3 / lambda1

    # Tensor de identidad
    I = ufl.Identity(2)

    # Calcular tensores de proyección individuales
    M1 = (C - lambda2 * I) / (lambda1 - lambda2)
    M2 = (C - lambda1 * I) / (lambda2 - lambda1)

    #lamU1 = sqrt(lambda1)
    #lamU2 = sqrt(lambda2)

    # Calcular tensor de deformación epsilon
    #d = ufl.ln(lamU1) * M1 + ufl.ln(lamU2) * M2
    d = 0.5*ufl.ln(lambda1) * M1 + 0.5*ufl.ln(lambda2) * M2
    epsilon= (d.T+d)*0.5
    return epsilon

def calcular_U(u):
    
    import numpy as np
    import ufl

    # Tensor de identidad
    I = ufl.Identity(2)
    # Tensor de deformación
    F = I + ufl.grad(u)
    # Tensor de deformación derecho o tensor de Cauchy-Green
    C = ufl.variable(F.T * F)
    
    # Tolerancia
    #tol = DOLFIN_EPS
    tol=0.000000001
    # Determinar perturbación a partir de la tolerancia
    pert = 2 * tol

    # Obtener invariantes requeridos
    I1 = ufl.tr(C)  # traza del tensor
    I3 = ufl.det(C)  # determinante del tensor

    # Valor preliminar para el discriminante
    D = I1**2 - 4 * I3
    # Agregar perturbación numérica al discriminante si está cerca de cero; asegurar positividad de D
    D = ufl.conditional(ufl.lt(D, tol), abs(D) + pert, D)

    lambda1 = 0.5 * (I1 + ufl.sqrt(D))
    lambda2 = I3 / lambda1

    # Tensor de identidad
    I = ufl.Identity(2)

    # Calcular tensores de proyección individuales
    M1 = (C - lambda2 * I) / (lambda1 - lambda2)
    M2 = (C - lambda1 * I) / (lambda2 - lambda1)

    lamU1 = ufl.sqrt(lambda1)
    lamU2 =ufl.sqrt(lambda2)

    # Calcular tensor de deformación epsilon
    #d = ufl.ln(lamU1) * M1 + ufl.ln(lamU2) * M2
    U = 0.5*(lamU1) * M1 + 0.5*(lamU1) * M2
   
    return U



def resolver_problema_elastico(E,nu,theta, Nmesh, L, Aprox="Finite", Modelo="Hyper"):
    import numpy as np
    import fenics as fe
    import ufl
    # Definir el cuadrado
    mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L, L), Nmesh, Nmesh, diagonal="crossed")

    # Definir el espacio de funciones
    V = fe.VectorFunctionSpace(mesh, "CG", 1, dim=2)

    # Definir las expresiones de rotación para cada esquina
    rot_topL = fe.Expression(("x[0]*cos(theta)-x[1]*sin(theta)","x[0]*sin(theta)+x[1]*cos(theta)-x[1]"), theta=theta, degree=0)
    rot_topR = fe.Expression(("x[0]*cos(theta)-x[1]*sin(theta)-x[0]","x[0]*sin(theta)+x[1]*cos(theta)-x[1]"), theta=theta, degree=0)
    rot_botR = fe.Expression(("x[0]*cos(theta)-x[1]*sin(theta)-x[0]","x[0]*sin(theta)+x[1]*cos(theta)"), theta=theta, degree=0)

    # Aplicar condiciones de contorno con rotación en cada esquina
    corner_fixed = fe.AutoSubDomain(lambda x: np.logical_and(fe.near(x[0], 0.0), fe.near(x[1], 0.0)))
    corner_topL = fe.AutoSubDomain(lambda x: np.logical_and(fe.near(x[0], 0.0), fe.near(x[1], L)))
    corner_topR = fe.AutoSubDomain(lambda x: np.logical_and(fe.near(x[0], L), fe.near(x[1], L)))
    corner_botR = fe.AutoSubDomain(lambda x: np.logical_and(fe.near(x[0], L), fe.near(x[1], 0.0)))

    bc_fixed_corner = fe.DirichletBC(V, fe.Constant((0., 0.)), corner_fixed, method='pointwise')  
    bc_topL = fe.DirichletBC(V, rot_topL, corner_topL, method='pointwise') 
    bc_topR = fe.DirichletBC(V, rot_topR, corner_topR, method='pointwise') 
    bc_botR = fe.DirichletBC(V, rot_botR, corner_botR, method='pointwise') 

    bcs = [bc_fixed_corner, bc_topL, bc_topR, bc_botR]

    # Definir las variables
    u = fe.Function(V, name="Displacement")  # Desplazamiento
    v = fe.TestFunction(V)  # Función de prueba

    f = fe.Constant((0, 0))  # Fuerza de corte
    T = fe.Constant((0, 0))  # Tensión de corte

    # Definir las propiedades del material hiperelástico
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    I = fe.Identity(2)  # Tensor de identidad
    F = I + fe.grad(u)  # Tensor de deformación
    C = fe.variable(F.T * F)  # Tensor de deformación derecho o tensor de Cauchy-Green

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")

    # Definir la energía potencial de deformación
    W = 0.5 * fe.inner(S, eps) * fe.dx - fe.dot(T, v) * fe.ds - fe.dot(f, v) * fe.dx 

    # Definir la forma residual de la ecuación no lineal
    F = fe.derivative(W, u, v)

    # Resolver el problema de minimización de energía
    fe.solve(F == 0, u, bcs, solver_parameters={'newton_solver': {'maximum_iterations': 100}})

    return u
############################################################################################################################33
def resolver_problema_elastico_C(E,nu,D, Nmesh, L, Aprox="Finite", Modelo="Hyper"):
    import numpy as np
    import fenics as fe
    import ufl
    mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L, L), Nmesh, Nmesh, diagonal="crossed")

    # Definir el espacio de funciones
    V = fe.VectorFunctionSpace(mesh, "CG", 1, dim=2)

    # Definir condiciones de contorno
    def boundary_bottom(x, on_boundary):
        return on_boundary and fe.near(x[1], 0) 

    def boundary_top(x, on_boundary):
        return on_boundary and fe.near(x[1], L)

    bc_bottom = fe.DirichletBC(V, fe.Constant((0, 0)), boundary_bottom)
    bc_top = fe.DirichletBC(V, fe.Constant((D,0)), boundary_top)  # Desplazamiento en x = 1 en la parte superior
    bcs = [bc_bottom, bc_top]

    # Definir las variables
    u = fe.Function(V)  # Desplazamiento
    v = fe.TestFunction(V)  # Función de prueba

    f = fe.Constant((0, 0))  # Fuerza de corte
    T = fe.Constant((0, 0))  # Tensión de corte

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    I = fe.Identity(2)  # Tensor de identidad
    F = I + fe.grad(u)  # Tensor de deformación
    C = F.T * F  # Tensor de deformación derecho o tensor de Cauchy-Green
    J = fe.det(F)

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")

    # Definir la energía potencial de deformación
    W = 0.5*fe.inner(S, eps) * fe.dx - fe.dot(T, v) * fe.ds - fe.dot(f, v) * fe.dx 

    # Definir la forma residual de la ecuación no lineal
    F = fe.derivative(W, u, v)

    # Resolver el problema de minimización de energía
    fe.solve(F == 0, u,bcs, solver_parameters={'newton_solver': {'maximum_iterations': 100}})

    return u

##########################################################################################################################################################################
def resolver_problema_elastico_T(E,nu,D, Nmesh, L, Aprox="Finite", Modelo="Hyper"):
    import numpy as np
    import fenics as fe
    import ufl
    mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L, L), Nmesh, Nmesh, diagonal="crossed")

    # Definir el espacio de funciones
    V = fe.VectorFunctionSpace(mesh, "CG", 1, dim=2)

    # Definir condiciones de contorno
    def boundary_bottom(x, on_boundary):
        return on_boundary and fe.near(x[1], 0) 

    def boundary_top(x, on_boundary):
        return on_boundary and fe.near(x[1], L)

    bc_bottom = fe.DirichletBC(V, fe.Constant((0, 0)), boundary_bottom)
    bc_top = fe.DirichletBC(V, fe.Constant((0,D)), boundary_top)  # Desplazamiento en x = 1 en la parte superior
    bcs = [bc_bottom, bc_top]

    # Definir las variables
    u = fe.Function(V)  # Desplazamiento
    v = fe.TestFunction(V)  # Función de prueba

    f = fe.Constant((0, 0))  # Fuerza de corte
    T = fe.Constant((0, 0))  # Tensión de corte

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    I = fe.Identity(2)  # Tensor de identidad
    F = I + fe.grad(u)  # Tensor de deformación
    C = F.T * F  # Tensor de deformación derecho o tensor de Cauchy-Green
    J = fe.det(F)

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps  # Tensor de esfuerzo de Piola-Kirchhoff
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")

    # Definir la energía potencial de deformación
    W = 0.5*fe.inner(S, eps) * fe.dx - fe.dot(T, v) * fe.ds - fe.dot(f, v) * fe.dx 

    # Definir la forma residual de la ecuación no lineal
    F = fe.derivative(W, u, v)

    # Resolver el problema de minimización de energía
    fe.solve(F == 0, u,bcs, solver_parameters={'newton_solver': {'maximum_iterations': 100}})

    return u
###############################################################################################################################33
def calcular_tension_efectiva(u,E,nu,Aprox, Modelo):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")
    return fe.inner(S, S)**0.5


def calcular_deformacion_efectiva(u,E,nu,Aprox, Modelo):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")
    return  fe.inner(eps, eps)**0.5

def calcular_energia_desformacion(u, E, nu, Aprox, Modelo):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if Aprox == "Finite":
        if Modelo == "Lineal":
            eps = fe.sym(fe.grad(u))
        elif Modelo=="Hyper":
            #Coordenadas Lagrangianas
            R=calcular_U(u)*fe.inv(F)
            eps = R*fe.sym(fe.grad(u))*R.T
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")

    return 0.5 * fe.inner(eps, S)
######################################################################################33333

def calcular_tension_YY(u, E, nu, Aprox, Modelo):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if Aprox == "Finite":
        eps = fe.sym(fe.grad(u))
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")
    
    # Componente YY de la tensión
    S_YY = S[1, 1]
    
    return S_YY

def calcular_deformacion_YY(u, E, nu, Aprox):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)

    if Aprox == "Finite":
        eps = fe.sym(fe.grad(u))
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")
    
    # Componente YY de la deformación
    eps_YY = eps[1, 1]
    
    return eps_YY

def calcular_tension_XY(u, E, nu, Aprox, Modelo):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if Aprox == "Finite":
        eps = fe.sym(fe.grad(u))
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")

    if Modelo == "Lineal":
        S = lmbda * fe.tr(eps) * I + 2.0 * mu * eps
    elif Modelo == "Hyper":
        J = fe.det(F)
        S = lmbda * fe.tr(eps)/J * I + 2.0 * (mu-ufl.ln(J))/J * eps
    else:
        raise ValueError("Modelo no coincide con ninguna condición válida.")
    
    # Componente YY de la tensión
    S_YY = S[0, 1]
    
    return S_YY

def calcular_deformacion_XY(u, E, nu, Aprox):
    import numpy as np
    import fenics as fe
    import ufl
    I = fe.Identity(2)
    F = I + fe.grad(u)
    C = fe.variable(F.T * F)

    if Aprox == "Finite":
        eps = fe.sym(fe.grad(u))
    elif Aprox == "Green":
        eps = 0.5 * (C - I)
    elif Aprox == "Henky":
        # Debes proporcionar una implementación adecuada de calcular_epsilon(u)
        eps = calcular_epsilon(u)
    else:
        raise ValueError("Aprox no coincide con ninguna condición válida.")
    
    # Componente YY de la deformación
    eps_YY = eps[0, 1]
    
    return eps_YY
