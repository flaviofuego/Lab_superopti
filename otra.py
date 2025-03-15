import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import flet as ft
from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")

def main(page: ft.Page):
    page.title = "Home"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window.height = 812
    page.window.width = 375
    page.bgcolor = "#a5d8ff"
    page.theme_mode = ft.ThemeMode.LIGHT
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    
    
    # Definir la función de costo
    def costo(x, y):
        return (x - 2)**2 + (y - 3)**2

    # Función para graficar la región factible con colores basados en el costo y marcar el máximo
    def plot_region_max(c=10, x_min=0, y_min=0):
        # Crear un grid de puntos
        x = np.linspace(x_min, c, 400)
        y = np.linspace(y_min, c, 400)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar la función de costo en cada punto del grid
        Z = costo(X, Y)
        
        # Definir la región factible: x>=x_min, y>=y_min y x+y<=c
        factible = (X >= x_min) & (Y >= y_min) & (X + Y <= c)
        
        # Para los puntos no factibles, asignamos NaN para que no se coloreen
        Z_feasible = np.where(factible, Z, np.nan)
        
        # Calcular el máximo en la región factible
        max_idx = np.nanargmax(Z_feasible)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z_feasible.flatten()
        x_max_val = X_flat[max_idx]
        y_max_val = Y_flat[max_idx]
        max_val = Z_flat[max_idx]
        
        # Graficar la región factible coloreada por el costo
        fig, axs = plt.subplots()
        im = axs.imshow(Z_feasible, extent=(x_min, c, y_min, c),
                            origin='lower', cmap='Greys', interpolation='nearest')
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_title('Región Factible (color ~ costo) con Máximo')
        
        # Agregar la barra de color
        cbar = plt.colorbar(im, ax=axs)
        cbar.set_label('Costo (mayor = más oscuro)')
        
        # Marcar el máximo en la gráfica
        axs.scatter(x_max_val, y_max_val, color='blue', marker='*', s=150,
                        label=f'Máximo: ({x_max_val:.2f}, {y_max_val:.2f})\nCosto = {max_val:.2f}')
        axs.legend()
        
        fig.tight_layout()
        return fig

    # Función para evaluar la función de costo en un punto dado
    def evaluar_punto(x, y):
        valor = costo(x, y)
        print(f"f({x}, {y}) = {valor}")

    # Ejemplo de uso:
    fig = plot_region_max(c=10)
    #evaluar_punto(4, 5)

    """ dt = 0.01
    t = np.arange(0, 30, dt)
    nse1 = np.random.randn(len(t))  # white noise 1
    nse2 = np.random.randn(len(t))  # white noise 2

    # Two signals with a coherent part at 10Hz and a random part
    s1 = np.sin(2 * np.pi * 10 * t) + nse1
    s2 = np.sin(2 * np.pi * 10 * t) + nse2

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, s1, t, s2)
    axs[0].set_xlim(0, 2)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("s1 and s2")
    axs[0].grid(True)

    cxy, f = axs[1].cohere(s1, s2, 256, 1.0 / dt)
    axs[1].set_ylabel("coherence")

    fig.tight_layout() """
    page.add(
        ft.Container(
            content=ft.Column(
                controls=[
                    
                    ft.Text("Lab 1", size=30),
                    ft.OutlinedButton(
                        text="Punto 1",
                        on_click=lambda e: page.go("/login"),
                        width=200,
                        height=60,
                        style=ft.ButtonStyle(
                            color="#ffffff",
                            bgcolor="#28a745",
                        ),
                    ),
                    ft.Text("or", size=20),
                    ft.OutlinedButton(
                        text="Punto 4",
                        on_click=lambda e: page.go("/sign"),
                        width=200,
                        height=60,
                        style=ft.ButtonStyle(
                            color="#ffffff",
                            bgcolor="#28a745",
                        ),
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
    )

    page.add(MatplotlibChart(fig, expand=True))

ft.app(main)