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

    dt = 0.01
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

    fig.tight_layout()
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