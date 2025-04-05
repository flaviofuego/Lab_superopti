# Importamos las librerias necesarias
import streamlit as st
import plotly.graph_objects as go
#Numeric libraries
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
import math
import pandas as pd

# Vectores de tiempo
Delta = 0.01  # Tiempo 1 Continua

# Tiempo para la señal continua 1
t1 = np.arange(-2, -1, Delta)
t2 = np.arange(-1, 1, Delta)
t3 = np.arange(1, 3, Delta)  # Corregido: de 1 a 3 es valor constante 3
t4 = np.arange(3, 4 + Delta, Delta)
t1_T = np.concatenate((t1, t2, t3, t4))  # Tiempo total

# Tiempo para la señal continua 2
t1_2 = np.arange(-3, -2, Delta)
t2_2 = np.arange(-2, 0, Delta)  # Combinando los tramos de -2 a -1 y -1 a 0
t3_2 = np.arange(0, 2, Delta)
t4_2 = np.arange(2, 3 + Delta, Delta)
t2_T = np.concatenate((t1_2, t2_2, t3_2, t4_2))  # Tiempo total

# Tiempo para secuencias discretas
n1 = np.arange(-5, 16 + 1)  # Secuencia discreta 1

n2_1 = np.arange(-10, -6 + 1)  # Secuencia discreta 2
n2_2 = np.arange(-5, 0 + 1)
n2_3 = np.arange(1, 5 + 1)
n2_4 = np.arange(6, 10 + 1)
n2 = np.concatenate((n2_1, n2_2, n2_3, n2_4))  # Tiempo total

# Declaración de Funciones y secuencias

# Función continua 1 según la descripción
x1_1 = 2 * t1 + 4  # Valores en [-2, -1], recta 2x+4
x1_2 = 2 * np.ones(len(t2))  # Valores en [-1, 1], constante 2
x1_3 = 3 * np.ones(len(t3))  # Valores en [1, 3], constante 3
x1_4 = -3 * t4 + 12  # Valores en [3, 4], recta -3x+12
x_t1 = np.concatenate((x1_1, x1_2, x1_3, x1_4))  # Funcion 1 Continua

# Función continua 2 según la descripción
x2_1 = t1_2 + 3  # Valores en [-3, -2], recta x+3
x2_2 = 0.5 * t2_2 + 3  # Valores en [-2, 0], recta (1/2)x+3
x2_3 = -t3_2 + 3  # Valores en [0, 2], recta -x+3
x2_4 = np.ones(len(t4_2))  # Valores en [2, 3], constante 1
# El último punto (x=3) se define como 0 automáticamente
x_t2 = np.concatenate((x2_1, x2_2, x2_3, x2_4))  # Funcion 2 continua

# Secuencia Discreta 1 según el laboratorio
x_n = [0, 0, 0, 0, 0, -4, 0, 3, 5, 2, -3, -1, 3, 6, 8, 3, -1, 0, 0, 0, 0, 0]

# Secuencia Discreta 2 según el laboratorio
x_n2_1 = np.zeros(len(n2_1))  # Valores en [-10, -6] = 0

x_n2_2 = np.zeros(len(n2_2))
for j in range(len(n2_2)):
    x_n2_2[j] = (3/4) ** (n2_2[j])  # Valores en [-5, 0] = (3/4)^n

x_n2_3 = np.zeros(len(n2_3))
for j in range(len(n2_3)):
    x_n2_3[j] = (7/4) ** (n2_3[j])  # Valores en [1, 5] = (7/4)^n

x_n2_4 = np.zeros(len(n2_4))  # Valores en [6, 10] = 0

x_n2 = np.concatenate((x_n2_1, x_n2_2, x_n2_3, x_n2_4))  # Secuencia Discreta 2


# Metodo 1 Tiempo Continuo: Desplazamiento/Escalamiento
def metodo1(t, f, a, t0):
    x = sp.Symbol("x")
    t1 = t - t0  # Desplazamiento temporal
    tesc = t1 / a  # Escalamiento temporal

    # Define un rango común para el eje x
    x_min = min(t1.min(), tesc.min(), t.min())
    x_max = max(t1.max(), tesc.max(), t.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=f, mode="lines", line=dict(color="blue"), name=f"Señal ({x})")
    )
    fig.update_layout(
        title=f"Señal Original: ({x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Gráfico 1: Señal desplazada en el tiempo
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=t1,
            y=f,
            mode="lines",
            name=f"Señal Desplazada ({x - t0})",
            line=dict(color="green"),
        )
    )
    fig1.update_layout(
        title=f"Señal Desplazada ({x - t0})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)

    # Gráfico 2: Señal escalada y desplazada en el tiempo
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=tesc,
            y=f,
            mode="lines",
            name=f"Señal Desplazada y Escalada ({a*(x-t0)})",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title=f"Señal Desplazada y Escalada: ({a*(x-t0)})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)

    # Mostrar las tres graficas
    with st.container():
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


# Metodo 2 Tiempo Continuo: Escalamiento/Desplazamiento
def metodo2(t, f, a, t0):
    x = sp.Symbol("x")
    tesc = t / a  # Escalamiento temporal
    t1 = tesc - (t0 / a)  # Desplazamiento temporal
    
    # Define un rango común para el eje x
    x_min = min(t1.min(), tesc.min(), t.min())
    x_max = max(t1.max(), tesc.max(), t.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=f, mode="lines", line=dict(color="blue"), name=f"Señal ({x})")
    )
    fig.update_layout(
        title=f"Señal Original: ({x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Gráfico 1: Señal escalada en el tiempo
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=tesc,
            y=f,
            mode="lines",
            name=f"Señal Escalada ({a*x})",
            line=dict(color="green"),
        )
    )
    fig1.update_layout(
        title=f"Señal Escalada ({a*x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)

    # Gráfico 2: Señal desplazada y escalada
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=t1,
            y=f,
            mode="lines",
            name=f"Señal Escalada y Desplazada ({a*(x-t0)})",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title=f"Señal Escalada y Desplazada: ({a*(x-t0)})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)

    # Mostrar las tres graficas
    with st.container():
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


# Función para gráfico de stem (señales discretas)
def stem(n, f, title, color):
    fig = go.Figure()

    # Añadir las líneas de los "stems"
    for x_val, y_val in zip(n, f):
        fig.add_trace(
            go.Scatter(
                x=[x_val, x_val],  # misma coordenada x
                y=[0, y_val],  # desde el eje hasta el valor en y
                mode="lines",  # solo dibuja la línea
                line=dict(color=color, dash="dash"),
                showlegend=False,
            )
        )

    # Añadir los marcadores
    fig.add_trace(
        go.Scatter(
            x=n,
            y=f,
            mode="markers",
            marker=dict(color=color, size=10),
            name=title,
        )
    )

    # Actualizar layout del gráfico
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=n),
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)


# Metodo 1 Discreto: Desplazamiento/Escalamiento
def metodo1D(n, f, M, n0):
    stem(n, f, "Señal Original", "yellow")
    if abs(M) < 1:  # Caso de interpolación
        n1 = n - n0  # Secuencia desplazada

        # Gráfico 1 - Secuencia Desplazada
        stem(n1, f, "Secuencia Desplazada", "green")
        
        # Factor de interpolación
        Z = int(1 / abs(M))
        L_n = len(f)
        
        # Crear un nuevo eje de tiempo para la interpolación
        nI = np.arange(n1[0] * Z, (n1[-1] * Z) + 1)
        L_nI = len(nI)
        
        # Inicializar arreglos para los diferentes métodos de interpolación
        x_nI0 = np.zeros(L_nI)  # Interpolación por cero
        x_nIEsc = np.zeros(L_nI)  # Interpolación escalón
        x_nM = np.zeros(L_nI)  # Interpolación lineal

        # Interpolación por cero y escalón
        for k in range(L_nI):
            if k % Z == 0:
                r = int(k * abs(M))
                if r < L_n:
                    x_nI0[k] = f[r]
                    x_nIEsc[k] = f[r]
            else:
                x_nI0[k] = 0
                if k > 0:
                    x_nIEsc[k] = x_nIEsc[k - 1]

        # Interpolación lineal
        k = 0
        for s in range(L_n - 1):
            x_nM[k] = f[s]
            for j in range(1, Z):
                if k + j < L_nI:
                    dif = f[s + 1] - f[s]
                    A = j * abs(M) * dif + f[s]
                    x_nM[k + j] = A
            k += Z
        
        # Asegurar que el último punto se asigna correctamente
        if k < L_nI:
            x_nM[k:] = f[-1]

        # Graficar según el signo de M
        if M > 0:
            # Gráfico 2 - Interpolación por cero
            stem(nI, x_nI0, "Interpolación por cero", "red")

            # Gráfico 3 - Interpolación escalón
            stem(nI, x_nIEsc, "Interpolación escalón", "blue")

            # Gráfico 4 - Interpolación lineal
            stem(nI, x_nM, "Interpolación lineal", "orange")
        else:
            # Invertir el eje temporal para M negativo
            nI_inv = -nI
            
            # Gráfico 2 - Interpolación por cero (invertido)
            stem(nI_inv, x_nI0, "Interpolación por cero", "red")

            # Gráfico 3 - Interpolación escalón (invertido)
            stem(nI_inv, x_nIEsc, "Interpolación escalón", "blue")

            # Gráfico 4 - Interpolación lineal (invertido)
            stem(nI_inv, x_nM, "Interpolación lineal", "orange")
    
    else:  # Caso de diezmado
        n1 = n - n0  # Secuencia desplazada
        
        # Gráfico 1 - Secuencia Desplazada
        stem(n1, f, "Secuencia Desplazada", "green")
        
        # Obtener puntos para el diezmado
        new_n = []
        new_f = []
        
        for i in range(len(n1)):
            if n1[i] % abs(M) == 0:
                new_n.append(n1[i] / abs(M))
                new_f.append(f[i])
                
        # Graficar según el signo de M
        if M > 0:
            # Gráfico 2 - Secuencia Diezmada
            stem(new_n, new_f, "Secuencia Diezmada", "red")
        else:
            # Invertir el eje temporal para M negativo
            new_n_inv = [-x for x in new_n]
            
            # Gráfico 2 - Secuencia Diezmada (invertida)
            stem(new_n_inv, new_f, "Secuencia Diezmada (invertida)", "red")


# Metodo 2 Discreto: Escalamiento/Desplazamiento
def metodo2D(n, f, M, n0):
    stem(n, f, "Señal Original", "yellow")
    
    if abs(M) < 1:  # Caso de interpolación
        # Factor de interpolación
        Z = int(1 / abs(M))
        L_n = len(f)
        
        # Crear un nuevo eje de tiempo para la interpolación
        nI = np.arange(n[0] * Z, (n[-1] * Z) + 1)
        L_nI = len(nI)
        
        # Inicializar arreglos para los diferentes métodos de interpolación
        x_nI0 = np.zeros(L_nI)  # Interpolación por cero
        x_nIEsc = np.zeros(L_nI)  # Interpolación escalón
        x_nM = np.zeros(L_nI)  # Interpolación lineal

        # Interpolación por cero y escalón
        for k in range(L_nI):
            if k % Z == 0:
                r = int(k * abs(M))
                if r < L_n:
                    x_nI0[k] = f[r]
                    x_nIEsc[k] = f[r]
            else:
                x_nI0[k] = 0
                if k > 0:
                    x_nIEsc[k] = x_nIEsc[k - 1]

        # Interpolación lineal
        k = 0
        for s in range(L_n - 1):
            x_nM[k] = f[s]
            for j in range(1, Z):
                if k + j < L_nI:
                    dif = f[s + 1] - f[s]
                    A = j * abs(M) * dif + f[s]
                    x_nM[k + j] = A
            k += Z
        
        # Asegurar que el último punto se asigna correctamente
        if k < L_nI:
            x_nM[k:] = f[-1]

        # Gráfico 1 - Interpolación por cero
        stem(nI, x_nI0, "Interpolación por cero", "red")

        # Gráfico 2 - Interpolación escalón
        stem(nI, x_nIEsc, "Interpolación escalón", "blue")

        # Gráfico 3 - Interpolación lineal
        stem(nI, x_nM, "Interpolación lineal", "orange")

        # Desplazamiento después de la interpolación
        if M > 0:  # Interpolación positiva
            fc = n0 / abs(M)
            nI_desp = nI - fc
            
            # Gráfico 4 - Interpolación por cero (desplazada)
            stem(nI_desp, x_nI0, "Interpolación por cero (desplazada)", "pink")

            # Gráfico 5 - Interpolación escalón (desplazada)
            stem(nI_desp, x_nIEsc, "Interpolación escalón (desplazada)", "purple")

            # Gráfico 6 - Interpolación lineal (desplazada)
            stem(nI_desp, x_nM, "Interpolación lineal (desplazada)", "brown")
        else:  # Interpolación negativa
            fc = n0 / abs(M)
            nI_desp = nI - fc
            nI_desp_inv = -nI_desp  # Invertir para M negativo
            
            # Gráfico 4 - Interpolación por cero (desplazada e invertida)
            stem(nI_desp_inv, x_nI0, "Interpolación por cero (desplazada)", "pink")

            # Gráfico 5 - Interpolación escalón (desplazada e invertida)
            stem(nI_desp_inv, x_nIEsc, "Interpolación escalón (desplazada)", "purple")

            # Gráfico 6 - Interpolación lineal (desplazada e invertida)
            stem(nI_desp_inv, x_nM, "Interpolación lineal (desplazada)", "brown")
    
    else:  # Caso de diezmado
        # Escalar primero
        new_n = []
        new_f = []
        
        for i in range(len(n)):
            if n[i] % abs(M) == 0:
                new_n.append(n[i] / abs(M))
                new_f.append(f[i])
        
        # Gráfico 1 - Secuencia Escalada (diezmada)
        if M > 0:
            stem(new_n, new_f, "Secuencia Escalada", "green")
        else:
            new_n_inv = [-x for x in new_n]
            stem(new_n_inv, new_f, "Secuencia Escalada (invertida)", "green")
        
        # Desplazar después de escalar
        new_n_desp = [x - (n0/abs(M)) for x in new_n]
        
        # Gráfico 2 - Secuencia Escalada y Desplazada
        if M > 0:
            stem(new_n_desp, new_f, "Secuencia Escalada y Desplazada", "blue")
        else:
            new_n_desp_inv = [-x for x in new_n_desp]
            stem(new_n_desp_inv, new_f, "Secuencia Escalada y Desplazada (invertida)", "blue")


# Operaciones específicas para punto 3: x(t/3 + 2) + x(1 - t/4)
def suma_continua(t, f):
    # Crear una función de interpolación para la señal original
    interp_f = interp1d(t, f, kind="linear", fill_value=0, bounds_error=False)
    
    # Calcular los argumentos transformados
    t1_arg = t/3 + 2  # Para x(t/3 + 2)
    t2_arg = 1 - t/4  # Para x(1 - t/4)
    
    # Calcular valores de las señales transformadas
    f1 = interp_f(t1_arg)
    f2 = interp_f(t2_arg)
    
    # Crear un rango común para visualización
    t_min = min(t.min(), t1_arg.min(), t2_arg.min())
    t_max = max(t.max(), t1_arg.max(), t2_arg.max())
    t_common = np.linspace(t_min, t_max, 1000)
    
    # Interpolar las señales transformadas al rango común
    f1_common = interp_f(t1_arg)
    f2_common = interp_f(t2_arg)
    
    # Sumar las señales
    f_suma = f1_common + f2_common
    
    # Graficar
    col1, col2 = st.columns(2)
    
    # Primera columna: x(t/3 + 2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=f1, mode="lines", name="x(t/3 + 2)"))

        fig1.update_layout(
            title="x(t/3 + 2)",
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            showlegend=True,
        )

        fig1.update_xaxes(showgrid=True)
        fig1.update_yaxes(showgrid=True)

        # Mostrar gráfico
        st.plotly_chart(fig1, use_container_width=True)

    # Segunda columna: x(1 - t/4)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=t,
                y=f2,
                mode="lines",
                name="x(1 - t/4)",
                line=dict(color="green"),
            )
        )
        fig2.update_layout(
            title="x(1 - t/4)",
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            showlegend=True,
        )

        fig2.update_xaxes(showgrid=True)
        fig2.update_yaxes(showgrid=True)

        # Mostrar gráfico
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico de la suma
    fig_sum = go.Figure()
    fig_sum.add_trace(
        go.Scatter(
            x=t,
            y=f_suma,
            mode="lines",
            name="x(t/3 + 2) + x(1 - t/4)",
            line=dict(color="red"),
        )
    )

    fig_sum.update_layout(
        title="Suma: x(t/3 + 2) + x(1 - t/4)",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=False,
    )

    fig_sum.update_xaxes(showgrid=True)
    fig_sum.update_yaxes(showgrid=True)

    st.plotly_chart(fig_sum, use_container_width=True)


# Suma de señales discretas
def suma_discreta(n, f):
    # Secuencia 1: x[M1n - n0] donde M1 = 1/4, n0 = -3
    M1 = 1/4
    n0_1 = -3
    
    # Secuencia 2: x[M2n - n0] donde M2 = -1/3, n0 = 4
    M2 = -1/3
    n0_2 = 4
    
    # Interpolación para la secuencia 1
    Z1 = int(1 / abs(M1))
    L_n = len(f)
    nI1 = np.arange(min(n) * Z1, (max(n) * Z1) + 1)
    L_nI1 = len(nI1)
    
    # Interpolación lineal para la secuencia 1
    x_nM1 = np.zeros(L_nI1)
    k = 0
    for s in range(L_n - 1):
        x_nM1[k] = f[s]
        for j in range(1, Z1):
            if k + j < L_nI1:
                dif = f[s + 1] - f[s]
                A = j * abs(M1) * dif + f[s]
                x_nM1[k + j] = A
        k += Z1
    
    # Desplazamiento para la secuencia 1
    nI1_shifted = nI1 - (n0_1 / M1)
    
    # Interpolación para la secuencia 2
    Z2 = int(1 / abs(M2))
    nI2 = np.arange(min(n) * Z2, (max(n) * Z2) + 1)
    L_nI2 = len(nI2)
    
    # Interpolación lineal para la secuencia 2
    x_nM2 = np.zeros(L_nI2)
    k = 0
    for s in range(L_n - 1):
        x_nM2[k] = f[s]
        for j in range(1, Z2):
            if k + j < L_nI2:
                dif = f[s + 1]
                dif = f[s + 1] - f[s]
                A = j * abs(M2) * dif + f[s]
                x_nM2[k + j] = A
        k += Z2
    
    # Invertir y desplazar para la secuencia 2 (ya que M2 es negativo)
    nI2_shifted = -(nI2 - (n0_2 / M2))
    
    # Graficar las secuencias
    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de la secuencia 1
        stem(nI1_shifted, x_nM1, "x[(n/4)-3]", "green")

    with col2:
        # Gráfico de la secuencia 2
        stem(nI2_shifted, x_nM2, "x[4-(n/3)]", "red")
    
    # Crear un eje de tiempo común para la suma
    n_min = min(min(nI1_shifted), min(nI2_shifted))
    n_max = max(max(nI1_shifted), max(nI2_shifted))
    
    # Ajustar para que el rango sea entero
    n_min = math.floor(n_min)
    n_max = math.ceil(n_max)
    
    n_common = np.arange(n_min, n_max + 1)
    
    # Interpolar ambas secuencias al eje común
    interp1 = interp1d(nI1_shifted, x_nM1, kind='linear', bounds_error=False, fill_value=0)
    interp2 = interp1d(nI2_shifted, x_nM2, kind='linear', bounds_error=False, fill_value=0)
    
    # Evaluar en el eje común
    seq1_common = interp1(n_common)
    seq2_common = interp2(n_common)
    
    # Sumar las secuencias
    sum_seq = seq1_common + seq2_common
    
    # Graficar la suma
    stem(n_common, sum_seq, "Suma de las secuencias", "blue")


# Función para cargar y procesar archivos de señales muestreadas
def cargar_senales():
    st.subheader("Cargar señales muestreadas")
    
    # Permitir la carga de archivos
    col1, col2 = st.columns(2)
    
    with col1:
        archivo1 = st.file_uploader("Cargar señal 1 (2kHz)", type=["txt"])
        if archivo1 is not None:
            st.success("Archivo 1 cargado correctamente")
    
    with col2:
        archivo2 = st.file_uploader("Cargar señal 2 (2.2kHz)", type=["txt"])
        if archivo2 is not None:
            st.success("Archivo 2 cargado correctamente")
    
    if archivo1 is not None and archivo2 is not None:
        # Leer datos
        try:
            datos1 = np.loadtxt(archivo1)
            datos2 = np.loadtxt(archivo2)
            
            # Definir frecuencias de muestreo
            fs1 = 2000  # 2 kHz
            fs2 = 2200  # 2.2 kHz
            
            # Calcular el mínimo común múltiplo para el sobremuestreo
            fs_comun = np.lcm(2000, 2200)  # 22000 Hz
            
            # Crear vectores de tiempo
            t1 = np.arange(len(datos1)) / fs1
            t2 = np.arange(len(datos2)) / fs2
            
            # Graficar señales originales
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=t1, y=datos1, mode="lines", name="Señal 1 (2kHz)"))
                fig1.update_layout(
                    title="Señal 1 (2kHz)",
                    xaxis_title="Tiempo (s)",
                    yaxis_title="Amplitud",
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t2, y=datos2, mode="lines", name="Señal 2 (2.2kHz)", line=dict(color="green")))
                fig2.update_layout(
                    title="Señal 2 (2.2kHz)",
                    xaxis_title="Tiempo (s)",
                    yaxis_title="Amplitud",
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Crear vector de tiempo común para el sobremuestreo
            t_max = max(t1[-1], t2[-1])
            t_comun = np.arange(0, t_max, 1/fs_comun)
            
            # Interpolar las señales a la frecuencia común
            interp_1 = interp1d(t1, datos1, kind='linear', bounds_error=False, fill_value=0)
            interp_2 = interp1d(t2, datos2, kind='linear', bounds_error=False, fill_value=0)
            
            datos1_interp = interp_1(t_comun)
            datos2_interp = interp_2(t_comun)
            
            # Graficar señales sobremuestreadas
            st.subheader("Señales sobremuestreadas")
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=t_comun, y=datos1_interp, mode="lines", name="Señal 1 Sobremuestreada"))
                fig3.update_layout(
                    title="Señal 1 Sobremuestreada",
                    xaxis_title="Tiempo (s)",
                    yaxis_title="Amplitud",
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=t_comun, y=datos2_interp, mode="lines", name="Señal 2 Sobremuestreada", line=dict(color="green")))
                fig4.update_layout(
                    title="Señal 2 Sobremuestreada",
                    xaxis_title="Tiempo (s)",
                    yaxis_title="Amplitud",
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # Sumar las señales
            suma = datos1_interp + datos2_interp
            
            # Graficar la suma
            fig_suma = go.Figure()
            fig_suma.add_trace(go.Scatter(x=t_comun, y=suma, mode="lines", name="Suma de señales", line=dict(color="red")))
            fig_suma.update_layout(
                title="Suma de las señales sobremuestreadas",
                xaxis_title="Tiempo (s)",
                yaxis_title="Amplitud",
            )
            st.plotly_chart(fig_suma, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")
    else:
        st.info("Por favor, carga ambos archivos para procesar las señales.")


# Configuración de la página de Streamlit
st.set_page_config(layout="wide", page_title="Transformación de Señales")
st.title("Laboratorio de Transformación de Señales")
st.markdown("### Universidad del Norte - 2025-10")

# Menú lateral
st.sidebar.title("Menú de operaciones")
operation = st.sidebar.selectbox(
    "Tipo de Señal", ["Menú Inicial", "Continua", "Discreta", "Cargar Señales"]
)

if operation == "Menú Inicial":
    st.markdown("""
    # Bienvenido al Laboratorio de Transformación de Señales
    
    Esta aplicación permite visualizar y transformar diferentes tipos de señales en tiempo continuo y discreto.
    
    ## Características:
    
    - **Señales Continuas**: Visualiza y transforma dos tipos de señales continuas.
    - **Señales Discretas**: Visualiza y transforma dos tipos de secuencias discretas.
    - **Transformaciones**: Aplica operaciones de desplazamiento y escalamiento.
    - **Interpolación**: Visualiza diferentes métodos de interpolación.
    - **Carga de archivos**: Procesa señales desde archivos .txt con diferentes frecuencias de muestreo.
    
    Selecciona una opción del menú lateral para comenzar.
    """)

elif operation == "Continua":
    st.subheader("Transformación de Señales Continuas")
    
    signal = st.sidebar.radio("Señal", ["1", "2"])
    if signal == "1":
        x = t1_T
        y = x_t1
        st.write("Señal Continua 1 seleccionada")
    else:
        x = t2_T
        y = x_t2
        st.write("Señal Continua 2 seleccionada")

    sum_option = st.sidebar.radio(
        "Operación",
        ["Transformación", "Suma x(t/3 + 2) + x(1 - t/4)"],
        key="sum_continua",
    )
    
    if sum_option == "Transformación":
        method = st.sidebar.radio(
            "Método de Transformación",
            ["Desplazamiento/Escalamiento", "Escalamiento/Desplazamiento"],
            key="method_continua",
        )

        a = st.sidebar.select_slider(
            "Factor de escalamiento (a)",
            np.round(
                [
                    -5,
                    -4,
                    -3,
                    -2,
                    -1/2,
                    -1/3,
                    -1/4,
                    -1/5,
                    1/5,
                    1/4,
                    1/3,
                    1/2,
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                2,
            ),
            value=2,
            key="a",
        )
        
        t0 = st.sidebar.select_slider(
            "Desplazamiento (t0)",
            [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
            value=1,
        )

        if method == "Desplazamiento/Escalamiento":
            metodo1(x, y, a, t0)
        else:
            metodo2(x, y, a, t0)
    else:
        suma_continua(x, y)

elif operation == "Discreta":
    st.subheader("Transformación de Señales Discretas")
    
    signal = st.sidebar.radio("Señal", ["1", "2"])
    if signal == "1":
        x = n1
        y = x_n
        st.write("Secuencia Discreta 1 seleccionada")
    else:
        x = n2
        y = x_n2
        st.write("Secuencia Discreta 2 seleccionada")

    sum_option = st.sidebar.radio(
        "Operación",
        ["Transformación", "Suma"],
        key="sum_discreto",
    )
    
    if sum_option == "Transformación":
        method = st.sidebar.radio(
            "Método de Transformación",
            ["Desplazamiento/Escalamiento", "Escalamiento/Desplazamiento"],
            key="method_discreta",
        )

        M = st.sidebar.select_slider(
            "Factor de escalamiento (M)",
            np.round(
                [
                    -4,
                    -3,
                    -2,
                    -1/2,
                    -1/3,
                    -1/4,
                    -1/5,
                    1/5,
                    1/4,
                    1/3,
                    1/2,
                    2,
                    3,
                    4,
                ],
                2,
            ),
            value=2,
        )
        
        n0 = st.sidebar.select_slider(
            "Desplazamiento (n0)",
            [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
            value=1,
        )

        if method == "Desplazamiento/Escalamiento":
            metodo1D(x, y, M, n0)
        else:
            metodo2D(x, y, M, n0)
    else:
        suma_discreta(x, y)

elif operation == "Cargar Señales":
    cargar_senales()