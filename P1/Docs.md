estado: ($\subset \mathbb{Z}$)
 - x: del 0 al 100 y -1 significa que no lo ve y la ultima vez que lo vio fue la izquierda y 101 la derecha.
 - y: del 0 al 100 y -1 significa que no lo ve
 - tamaño blob:


acciones ($\subset \mathbb{R}^2$):
 - moveWheelsL 
 - moveWheelsR 
 - NOTA: para normalizar, hacer (la parte de programacion) que permita no uniforme

función recompensa:
 - $$\alpha_1  e^{-d^2} + \alpha_2 e^{-(x-50)^2}$$
 donde $d$ es distancia y $x$ es $x$ . 
 - NOTA: debe ser modular probar más
