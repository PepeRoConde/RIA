from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color






robocop = Robobo("localhost")
robocop.connect()



robocop.moveWheelsByTime(-20,-20,5)
robocop.moveTiltTo(110,20)