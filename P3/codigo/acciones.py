from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import yaml 

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class AccionesRobobo:
    def __init__(self, robobo):
        self.robobo = robobo

    def derecha(self, velocidad=config['velocidad'], tiempo=config['tiempo']):
        self.robobo.moveWheelsByTime(0, velocidad, tiempo)

    def izquierda(self, velocidad=config['velocidad'], tiempo=config['tiempo']):
        self.robobo.moveWheelsByTime(velocidad, 0, tiempo)

    def adelante(self, velocidad=config['velocidad'], tiempo=config['tiempo']):
        self.robobo.moveWheelsByTime(velocidad, velocidad, tiempo)

    def atras(self, velocidad=config['velocidad'], tiempo=config['tiempo']):
        self.robobo.moveWheelsByTime(-velocidad, -velocidad, tiempo)

    def quieto(self):
        self.robobo.stopMotors()


def crear_robot():
    robocop = Robobo("localhost")
    robocop.connect()

    sim = RoboboSim("localhost")
    sim.connect()

    return AccionesRobobo(robocop)
