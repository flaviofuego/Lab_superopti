# Importamos las librerías necesarias
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# CSS personalizado (sin cambios en la estructura)
st.markdown(
    """
    <style>
    /* Fondo de la aplicación con Blanco */
    .stApp {
        background-color: #313232;
    }
    /* Fondo secundario - Cuadros de selección */
    .stselectbox .css-1d391kg {
        background-color: #ADD8E6;
    }
    /* Botones */
    .stButton>button {
        background-color: #ADD8E6;
        color: white;
        border-radius: 10px;
        border: none;
    }
    /* Botones al pasar el mouse */
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    /* Cajas de selección */
    .stSelectbox>div>div>div {
        background-color: #ADD8E6;
        color: black;
        border-radius: 6px;
    }
    /* Texto general */
    .stMarkdown, .stSidebar .css-1d391kg, h1, h2, h3, h4, label {
        color: #000000;
    }
    /* Caja de entrada de texto */
    .stTextInput>div>input {
        background-color: #4682B4;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#########################
# FUNCIONES CONTINUAS
#########################
def signal_1():
    delta = 0.01  # tiempo de muestreo
    t1 = np.arange(-2, -1, delta)
    t2 = np.arange(-1, 1, delta)
    t3 = np.arange(1, 3, delta)
    t4 = np.arange(3, 4 + delta, delta)
    x1 = 2 * t1 + 4
    x2 = 2 * np.ones(len(t2))
    x3 = 3 * np.ones(len(t3))
    x4 = -3 * t4 + 12
    t = np.concatenate((t1, t2, t3, t4))
    x_t = np.concatenate((x1, x2, x3, x4))
    return t, x_t

def signal_2():
    Delta2 = 0.01 
    t1_2 = np.arange(-3, -2 + Delta2, Delta2)
    t2_2 = np.arange(-2, 0 + Delta2, Delta2)
    t3_2 = np.arange(0, 2 + Delta2, Delta2)
    t4_2 = np.arange(2, 3, Delta2)
    t5_2 = np.arange(3, 3 + Delta2, Delta2)
    t_total = np.concatenate((t1_2, t2_2, t3_2, t4_2, t5_2))
    x2_1 = t1_2 + 3
    x2_2 = (1/2) * t2_2 + 3
    x2_3 = -t3_2 + 3
    x2_4 = np.ones(len(t4_2))
    x2_5 = [0]
    x_total = np.concatenate((x2_1, x2_2, x2_3, x2_4, x2_5))
    return t_total, x_total

def transform_señal_continua(t, x, t0, a1, method):
    if method == 'Metodo 1':
        # Primero desplazamos, luego escalamos
        t_transformed = (t - t0) / a1
    elif method == 'Metodo 2':
        # Primero escalamos, luego desplazamos
        t_transformed = (t / a1) - (t0 / a1)
    return t_transformed, x

#########################
# FUNCIONES DISCRETAS
#########################
def señal_discreta1():
    n = np.arange(-6, 17)  # 23 puntos
    x_n = np.array([
        0, 0, 0, 0, 0, 0,  # n = -6 a -1
        -4,                # n = 0
        0, 3, 5, 2, -3, -1, 3, 6, 8, 3, -1,  # n = 1 a 11
        0, 0, 0, 0, 0      # n = 12 a 16
    ], dtype=float)
    return n, x_n

def señal_discreta2():
    n2 = np.arange(-10, 11)
    x_n2 = np.zeros(len(n2))
    x_n2[(n2 >= -5) & (n2 <= 0)] = (3/4) ** n2[(n2 >= -5) & (n2 <= 0)]
    x_n2[(n2 >= 1) & (n2 <= 5)] = (7/4) ** n2[(n2 >= 1) & (n2 <= 5)]
    return n2, x_n2

def transform_señal_discreta(n, x_n, n0, M, method):
    if method == 'Metodo 1':
        n_transformed = (n - n0) / M
    elif method == 'Metodo 2':
        n_transformed = (n / M) - (n0 / M)
    return n_transformed, x_n

#########################
# SUMA DE SEÑALES CONTINUAS
#########################
def sumar_senales_continuas(t1, x1, t2, x2, desplazamiento1=0, escalamiento1=1, desplazamiento2=0, escalamiento2=1):
    t1_trans = (t1 + desplazamiento1) / escalamiento1
    t2_trans = (t2 + desplazamiento2) / escalamiento2
    interp_x1 = interp1d(t1_trans, x1, kind='linear', fill_value=0, bounds_error=False)
    interp_x2 = interp1d(t2_trans, x2, kind='linear', fill_value=0, bounds_error=False)
    min_total = min(t1_trans.min(), t2_trans.min())
    max_total = max(t1_trans.max(), t2_trans.max())
    t_comun = np.linspace(min_total, max_total, 1000)
    x1_interp = interp_x1(t_comun)
    x2_interp = interp_x2(t_comun)
    x_suma = x1_interp + x2_interp
    return t_comun, x_suma

#########################
# FUNCIONES DE INTERPOLACIÓN DISCRETA
#########################
def interpolacion_por_ceros(x, factor):
    N = len(x)
    L = N * factor
    y = np.zeros(L)
    for i in range(N):
        y[i * factor] = x[i]
    return y

def interpolacion_escalonada(x, factor):
    N = len(x)
    L = N * factor
    y = np.zeros(L)
    idx = 0
    for i in range(N):
        for _ in range(factor):
            y[idx] = x[i]
            idx += 1
    return y

def interpolacion_lineal(x, factor):
    N = len(x)
    L = (N - 1) * factor + 1
    y = np.zeros(L)
    pos = 0
    for i in range(N - 1):
        y[pos] = x[i]
        paso = (x[i+1] - x[i]) / factor
        for k in range(1, factor):
            y[pos + k] = x[i] + k * paso
        pos += factor
    y[-1] = x[-1]
    return y

# Función de ploteo tipo stem usando matplotlib (definida una sola vez)
def stem_plot(n, f, title, color):
    plt.figure()
    marker = color + 'o'
    plt.stem(n, f, linefmt=" ", markerfmt=marker, basefmt="black")
    plt.title(title)
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

#########################
# CONFIGURACIÓN DE LA PÁGINA
#########################
st.markdown("<h1 style='color: #FFFFFFFF;'>Operaciones de Señales continuas y discretas</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #FFFFFFFF;'>By: Jose Hernandez, Rosa Hernandez, Juan Acevedo</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #FFFFFFFF;'>Señales y Sistemas - Universidad del Norte - PhD. Juan P Tello</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #FFFFFFFF;'>2025</p>", unsafe_allow_html=True)

# Menú principal
main_menu = st.sidebar.selectbox('Seleccione una función principal', 
                                 ['Graficación de señales', 'Transformación de funciones', 'Suma de señales', 'Suma e interpolacion a diferentes frecuencia'])

##################################
# TRANSFORMACIÓN DE FUNCIONES
##################################
if main_menu == 'Transformación de funciones':
    transform_menu = st.sidebar.selectbox('Seleccione el tipo de función', ['Funciones continuas', 'Funciones discretas'])
    
    if transform_menu == 'Funciones continuas':
        st.subheader('Transformación de Funciones Continuas')
        signal_choice = st.sidebar.selectbox('Seleccione una señal', ['Señal 1', 'Señal 2'])
        method = st.sidebar.selectbox('Seleccione el método de transformación', ['Metodo 1', 'Metodo 2'])
        t0_sel = st.sidebar.select_slider('Desplazamiento (t0)', 
                                            [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        a1_sel = st.sidebar.select_slider('Escalamiento (a1)', 
                                          ['1', '2', '3', '4', '5', '-1', '-2', '-3', '-4', '-5', 
                                           '1/2', '1/3', '1/4', '1/5', '-1/2', '-1/3', '-1/4', '-1/5'])
        a1_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '-1': -1, '-2': -2, '-3': -3, '-4': -4, '-5': -5,
                   '1/2': 0.5, '1/3': 1/3, '1/4': 0.25, '1/5': 0.2, '-1/2': -0.5, '-1/3': -1/3, '-1/4': -0.25, '-1/5': -0.2}
        # Convertir las selecciones a valores numéricos
        t0 = float(a1_dict.get(str(t0_sel), t0_sel))
        a1 = a1_dict[a1_sel]
        
        if st.button('Transformar y Graficar'):
            if signal_choice == 'Señal 1':
                t, x = signal_1()
            else:
                t, x = signal_2()
            if method == "Metodo 1":
                # Primer desplazamos y luego escalamos: (t-t0)/a1
                t_desplazada, _ = transform_señal_continua(t, x, t0, 1, 'Metodo 1')
                t_transform = t_desplazada / a1
                # Graficamos señal original vs. desplazada y luego la transformación completa
                fig1, ax1 = plt.subplots()
                ax1.plot(t, x, label='Original', color='red')
                ax1.plot(t_desplazada, x, label='Desplazada', color='blue', linestyle='--')
                ax1.grid(True)
                ax1.legend()
                ax1.set_xlabel('Tiempo')
                ax1.set_ylabel('Amplitud')
                ax1.set_title('Señal Original vs. Desplazada')
                st.pyplot(fig1)
                
                fig2, ax2 = plt.subplots()
                ax2.plot(t_transform, x, label='Escalada (Total)', color='blue')
                ax2.plot(t_desplazada, x, label='Desplazada', color='green', linestyle='--')
                ax2.grid(True)
                ax2.legend()
                ax2.set_xlabel('Tiempo')
                ax2.set_ylabel('Amplitud')
                ax2.set_title('Señal Desplazada vs. Escalada')
                st.pyplot(fig2)
            elif method == "Metodo 2":
                # Primero escalamos y luego desplazamos: (t/a1) - (t0/a1)
                t_escalada = t / a1
                t_transform = t_escalada - (t0 / a1)
                fig3, ax3 = plt.subplots()
                ax3.plot(t, x, label='Original', color='blue')
                ax3.plot(t_escalada, x, label='Escalada', color='green', linestyle='--')
                ax3.grid(True)
                ax3.legend()
                ax3.set_xlabel('Tiempo')
                ax3.set_ylabel('Amplitud')
                ax3.set_title('Señal Original vs. Escalada')
                st.pyplot(fig3)
                
                fig4, ax4 = plt.subplots()
                ax4.plot(t_escalada, x, label='Escalada', color='green', linestyle='--')
                ax4.plot(t_transform, x, label='Desplazada', color='red')
                ax4.grid(True)
                ax4.legend()
                ax4.set_xlabel('Tiempo')
                ax4.set_ylabel('Amplitud')
                ax4.set_title('Señal Escalada vs. Desplazada')
                st.pyplot(fig4)
    
    elif transform_menu == 'Funciones discretas':
        st.subheader('Transformación de Funciones Discretas')
        discrete_choice = st.sidebar.selectbox('Seleccione una secuencia discreta', 
                                                 ['Secuencia Discreta 1', 'Secuencia Discreta 2'])
        method = st.sidebar.selectbox('Seleccione el método de transformación', ['Metodo 1', 'Metodo 2'])
        interpolation_method = st.sidebar.selectbox("Seleccione el método de interpolación",
                                                     ["Interpolación por ceros", "Interpolación lineal", "Interpolación escalonada"])
        factor_dict = {"2": 2, "3": 3, "4": 4, "1/2": 0.5, "1/3": 0.33, "1/4": 0.25, "1/5": 0.2}
        n_disp_dict = {"1": 1, "-1": -1, "3": 3, "-3": -3, "4": 4, "-4": -4, "5": 5, "-5": -5, "6": 6, "-6": -6}
        selected_factor = st.sidebar.select_slider('Factor de interpolación', list(factor_dict.keys()))
        select_n = st.sidebar.select_slider("Desplazamiento", list(n_disp_dict.keys()))
        factor_val = factor_dict[selected_factor]
        n0 = n_disp_dict[select_n]
        if factor_val >= 1:
            factor_int = int(round(factor_val))
        else:
            factor_int = int(round(1 / factor_val))
        
        if discrete_choice == 'Secuencia Discreta 1':
            n, x_n = señal_discreta1()
        else:
            n, x_n = señal_discreta2()
        
        if interpolation_method == 'Interpolación por ceros':
            x_int = interpolacion_por_ceros(x_n, factor_int)
        elif interpolation_method == 'Interpolación lineal':
            x_int = interpolacion_lineal(x_n, factor_int)
        else:
            x_int = interpolacion_escalonada(x_n, factor_int)
        
        L = len(x_int)
        # Construir el eje de la señal interpolada basado en el factor
        n_interpolado = np.arange(n[0] * factor_int, n[0] * factor_int + L)
        n_escalado = n_interpolado / factor_int
        n_escalado_desplazado = n_escalado - n0  # Usamos la variable n0
        
        if st.button('Transformar y Graficar'):
            if method == "Metodo 1":
                fig, axs = plt.subplots(1, 3, figsize=(25, 8))
                axs[0].stem(n, x_n, linefmt='b-', markerfmt='bo', basefmt='b-', label='Original')
                axs[0].grid(True)
                axs[0].set_title("Señal Original")
                axs[0].set_xlabel("n")
                axs[0].set_ylabel("Amplitud")
                axs[0].legend()
                axs[1].stem(n_escalado, x_int, linefmt='g-', markerfmt='go', basefmt='g-', label='Escalada')
                axs[1].grid(True)
                axs[1].set_title("Señal Escalada")
                axs[1].set_xlabel("n")
                axs[1].set_ylabel("Amplitud")
                axs[1].legend()
                axs[2].stem(n_escalado_desplazado, x_int, linefmt='r-', markerfmt='ro', basefmt='r-', label='Escalada y Desplazada')
                axs[2].grid(True)
                axs[2].set_title("Señal Escalada y Desplazada")
                axs[2].set_xlabel("n")
                axs[2].set_ylabel("Amplitud")
                axs[2].legend()
                plt.tight_layout()
                st.pyplot(fig)
            elif method == "Metodo 2":
                fig, axs = plt.subplots(1, 3, figsize=(25, 8))
                axs[0].stem(n, x_n, linefmt='b-', markerfmt='bo', basefmt='b-', label='Original')
                axs[0].grid(True)
                axs[0].set_title("Señal Original")
                axs[0].set_xlabel("n")
                axs[0].set_ylabel("Amplitud")
                axs[0].legend()
                axs[1].stem(n - n0, x_n, linefmt='m-', markerfmt='mo', basefmt='m-', label='Desplazada')
                axs[1].grid(True)
                axs[1].set_title("Señal Desplazada")
                axs[1].set_xlabel("n")
                axs[1].set_ylabel("Amplitud")
                axs[1].legend()
                axs[2].stem(n_escalado_desplazado, x_int, linefmt='r-', markerfmt='ro', basefmt='r-', label='Escalada y Desplazada')
                axs[2].grid(True)
                axs[2].set_title("Señal Escalada y Desplazada")
                axs[2].set_xlabel("n")
                axs[2].set_ylabel("Amplitud")
                axs[2].legend()
                plt.tight_layout()
                st.pyplot(fig)
                
##################################
# GRAFICACIÓN DE SEÑALES
##################################
elif main_menu == 'Graficación de señales':
    st.subheader('Graficación de Señales')
    transform_menu = st.sidebar.selectbox('Seleccione el tipo de función', ['Funciones continuas', 'Funciones discretas'])
    
    if transform_menu == 'Funciones continuas':
        signal_choice = st.sidebar.selectbox('Seleccione una señal', ['Señal 1', 'Señal 2'])
        if st.button('Graficar'):
            if signal_choice == 'Señal 1':
                t, x = signal_1()
            else:
                t, x = signal_2()
            fig, ax = plt.subplots()
            ax.plot(t, x)
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Amplitud")
            ax.grid(True)
            st.pyplot(fig)
    if transform_menu == 'Funciones discretas':
        signal_choice = st.sidebar.selectbox('Seleccione una secuencia', ['Secuencia Discreta 1', 'Secuencia Discreta 2'])
        if st.button('Graficar'):
            if signal_choice == 'Secuencia Discreta 1':
                n, x_n = señal_discreta1()
            else:
                n, x_n = señal_discreta2()
            fig, ax = plt.subplots()
            ax.stem(n, x_n)
            ax.grid(True)
            st.pyplot(fig)
            
##################################
# SUMA DE SEÑALES CONTINUAS
##################################
elif main_menu == 'Suma de señales':
    suma_menu = st.sidebar.selectbox('Seleccione el tipo de señal', ['Señales continuas'])
    
    if suma_menu == 'Señales continuas':
        st.subheader('Suma de Señales Continuas')
        signal_choice_1 = st.sidebar.selectbox('Seleccione la primera señal', ['Señal 1', 'Señal 2'])
        signal_choice_2 = st.sidebar.selectbox('Seleccione la segunda señal', ['Señal 1', 'Señal 2'])
        desplazamiento1_sel = st.sidebar.selectbox('Desplazamiento para la señal 1', ['2', '1', '-1/2'])
        desplazamiento2_sel = st.sidebar.selectbox('Desplazamiento para la señal 2', ['1', '2', '1/3'])
        escalamiento1_sel = st.sidebar.selectbox('Escalamiento para la señal 1', ['1/3', '-1/4', '1/2'])
        escalamiento2_sel = st.sidebar.selectbox('Escalamiento para la señal 2', ['-1/4', '1/3', '1/3', '1/2'])
        desplazamiento_dict = {'1': 1, '2': 2, '-1/2': -0.5, '1/3': 0.33}
        a1_dict = {'-1/4': -0.25, '1/3': 0.33, '1/2': 0.5}
        
        desplazamiento1 = desplazamiento_dict[desplazamiento1_sel]
        desplazamiento2 = desplazamiento_dict[desplazamiento2_sel]
        escalamiento1 = a1_dict[escalamiento1_sel]
        escalamiento2 = a1_dict[escalamiento2_sel]
        
        if st.button('Sumar y Graficar'):
            if signal_choice_1 == 'Señal 1':
                t1, x1 = signal_1()
            else:
                t1, x1 = signal_2()
            if signal_choice_2 == 'Señal 1':
                t2, x2 = signal_1()
            else:
                t2, x2 = signal_2()
            t_comun, x_suma = sumar_senales_continuas(t1, x1, t2, x2, desplazamiento1, escalamiento1, desplazamiento2, escalamiento2)
            fig, ax = plt.subplots()
            ax.plot(t_comun, x_suma, label='Señal Suma')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Amplitud")
            st.pyplot(fig)
            
##################################
# SUMA E INTERPOLACIÓN A DIFERENTES FRECUENCIAS
##################################
elif main_menu == "Suma e interpolacion a diferentes frecuencia":
    st.sidebar.header("Parámetros de Muestreo")
    fs1 = st.sidebar.number_input("Frecuencia de muestreo de la Señal 1 (Hz)", value=2000, step=100)
    fs2 = st.sidebar.number_input("Frecuencia de muestreo de la Señal 2 (Hz)", value=2200, step=100)
    fs_new = st.sidebar.number_input("Frecuencia de muestreo objetivo (Hz)", value=22000, step=1000)
    I1 = int(fs_new / fs1)
    I2 = int(fs_new / fs2)
    
    uploaded_file1 = st.file_uploader("Cargar archivo de Señal 1 (.txt)", type=["txt"], key="file1")
    uploaded_file2 = st.file_uploader("Cargar archivo de Señal 2 (.txt)", type=["txt"], key="file2")
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        signal1 = np.loadtxt(uploaded_file1)
        signal2 = np.loadtxt(uploaded_file2)
        t1 = np.arange(len(signal1)) / fs1
        t2 = np.arange(len(signal2)) / fs2
        
        fig_orig, ax_orig = plt.subplots(2, 1, figsize=(10, 6))
        ax_orig[0].plot(t1, signal1, color="blue", label="Señal 1")
        ax_orig[0].set_title(f"Señal 1 (fs = {fs1} Hz)")
        ax_orig[0].set_xlabel("Tiempo (s)")
        ax_orig[0].set_ylabel("Amplitud")
        ax_orig[0].grid(True)
        ax_orig[0].legend()
        ax_orig[1].plot(t2, signal2, color="orange", label="Señal 2")
        ax_orig[1].set_title(f"Señal 2 (fs = {fs2} Hz)")
        ax_orig[1].set_xlabel("Tiempo (s)")
        ax_orig[1].set_ylabel("Amplitud")
        ax_orig[1].grid(True)
        ax_orig[1].legend()
        plt.tight_layout()
        st.pyplot(fig_orig)
        
        # Funciones de interpolación
        def interp_ceros(x, I):
            L = len(x)
            L_new = L * I
            y = np.zeros(L_new)
            for i in range(L):
                y[i * I] = x[i]
            return y

        def interp_escalon(x, I):
            L = len(x)
            L_new = L * I
            y = np.zeros(L_new)
            for i in range(L):
                for k in range(I):
                    y[i * I + k] = x[i]
            return y

        def interp_lineal(x, I):
            L = len(x)
            L_new = (L - 1) * I + 1
            y = np.zeros(L_new)
            for i in range(L - 1):
                y[i * I] = x[i]
                paso = (x[i+1] - x[i]) / I
                for k in range(1, I):
                    y[i * I + k] = x[i] + paso * k
            y[-1] = x[-1]
            return y

        interp_method = st.sidebar.selectbox("Seleccione el método de interpolación", 
                                               ["Interpolación por ceros", "Interpolación escalonada", "Interpolación lineal"])
        
        if interp_method == "Interpolación por ceros":
            signal1_interp = interp_ceros(signal1, I1)
            signal2_interp = interp_ceros(signal2, I2)
        elif interp_method == "Interpolación escalonada":
            signal1_interp = interp_escalon(signal1, I1)
            signal2_interp = interp_escalon(signal2, I2)
        elif interp_method == "Interpolación lineal":
            signal1_interp = interp_lineal(signal1, I1)
            signal2_interp = interp_lineal(signal2, I2)
        
        t1_interp = np.arange(len(signal1_interp)) / fs_new
        t2_interp = np.arange(len(signal2_interp)) / fs_new
        
        min_len = min(len(signal1_interp), len(signal2_interp))
        sum_signal = signal1_interp[:min_len] + signal2_interp[:min_len]
        t_sum = np.arange(min_len) / fs_new
        
        fig_interp, ax_interp = plt.subplots(3, 1, figsize=(10, 10))
        ax_interp[0].plot(t1_interp, signal1_interp, color="blue", label="Señal 1 Interpolada")
        ax_interp[0].set_title(f"Señal 1 Interpolada (fs = {fs_new} Hz)")
        ax_interp[0].set_xlabel("Tiempo (s)")
        ax_interp[0].set_ylabel("Amplitud")
        ax_interp[0].grid(True)
        ax_interp[0].legend()
        ax_interp[1].plot(t2_interp, signal2_interp, color="orange", label="Señal 2 Interpolada")
        ax_interp[1].set_title(f"Señal 2 Interpolada (fs = {fs_new} Hz)")
        ax_interp[1].set_xlabel("Tiempo (s)")
        ax_interp[1].set_ylabel("Amplitud")
        ax_interp[1].grid(True)
        ax_interp[1].legend()
        ax_interp[2].plot(t_sum, sum_signal, color="green", label="Suma de Señales")
        ax_interp[2].set_title("Suma de Señales Interpoladas")
        ax_interp[2].set_xlabel("Tiempo (s)")
        ax_interp[2].set_ylabel("Amplitud")
        ax_interp[2].grid(True)
        ax_interp[2].legend()
        plt.tight_layout()
        st.pyplot(fig_interp)
    else:
        st.write("Por favor, cargue ambos archivos para continuar.")
