from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box


class UI:
    def __init__(self):
        self.paso = 0
        self.accion = [0, 0]
        self.origen = "?"
        self.recompensa = 0

        self._live = None

    def update(self, paso, accion, origen, recompensa):
        self.paso = paso
        self.accion = accion
        self.origen = origen
        self.recompensa = recompensa

        if self._live:
            self._live.update(self.render())

    def render(self):
        text = Text()
        text.append(f"-- Paso #{self.paso}\n", style="bold cyan")
        text.append(f"Acción: {self.accion}\n", style="white")
        text.append(f"Robot dirigido por: {self.origen}\n", style="magenta")
        text.append(f"Recompensa: {self.recompensa:.3f}", style="green")

        return Panel(text, box=box.ROUNDED, title="Simulación Robobo")

    def start(self):
        self._live = Live(self.render(), refresh_per_second=10, screen=False)
        return self._live

ui = UI()
