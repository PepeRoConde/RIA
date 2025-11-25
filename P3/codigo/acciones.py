from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

class AccionesRobobo:
    def __init__(self, robobo):
        self.robobo = robobo

    def derecha(self, velocidad=20, tiempo=0.04):
        self.robobo.moveWheelsByTime(0, velocidad, tiempo)

    def izquierda(self, velocidad=20, tiempo=0.04):
        self.robobo.moveWheelsByTime(velocidad, 0, tiempo)

    def adelante(self, velocidad=20, tiempo=0.04):
        self.robobo.moveWheelsByTime(velocidad, velocidad, tiempo)

    def atras(self, velocidad=20, tiempo=0.04):
        self.robobo.moveWheelsByTime(-velocidad, -velocidad, tiempo)

    def quieto(self):
        self.robobo.stopMotors()


def crear_robot():
    robocop = Robobo("localhost")
    robocop.connect()

    sim = RoboboSim("localhost")
    sim.connect()

    return AccionesRobobo(robocop)
