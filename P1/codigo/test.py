from typing import Optional
import time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import math

robocop = Robobo("localhost")
robocop.connect()
sim = RoboboSim("localhost")
sim.connect()

objetos = sim.getObjects()
if objetos != None and len(objetos) ==1 :
    for objeto in objetos:
        posicion = sim.getObjectLocation(objeto)['position']
        x_obj, y_obj = posicion['x'], posicion['y']

print(f' objeto ({x_obj},{y_obj})')
            
