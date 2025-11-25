import numpy as np

def detectar_posicion_brazos(keypoints):
    if len(keypoints) < 11:
        return "SIN BRAZOS"
    
    keypoints = np.array(keypoints)
    
    hombro_izq = keypoints[5]
    hombro_der = keypoints[6]
    muñeca_izq = keypoints[9]
    muñeca_der = keypoints[10]
    codo_izq = keypoints[7]
    codo_der = keypoints[8]
    
    brazos_detectados = {
        'izq': muñeca_izq[0] > 0 and muñeca_izq[1] > 0,
        'der': muñeca_der[0] > 0 and muñeca_der[1] > 0
    }
    
    if not brazos_detectados['izq'] and not brazos_detectados['der']:
        return "SIN BRAZOS"

    altura_hombro_izq = hombro_izq[1]
    altura_hombro_der = hombro_der[1]

    altura_muñeca_izq = muñeca_izq[1]
    altura_muñeca_der = muñeca_der[1]

    x_muñeca_izq = muñeca_izq[0]
    x_muñeca_der = muñeca_der[0]

    distancia_muñecas = abs(x_muñeca_der - x_muñeca_izq)

    brazo_izq_arriba = altura_muñeca_izq < altura_hombro_izq - 100
    brazo_der_arriba = altura_muñeca_der < altura_hombro_der - 100
    
    brazo_izq_abajo = altura_muñeca_izq > altura_hombro_izq + 150
    brazo_der_abajo = altura_muñeca_der > altura_hombro_der + 150
    
    if (brazo_izq_arriba and brazo_der_arriba and 
        distancia_muñecas > 200 and
        abs(altura_muñeca_izq - altura_muñeca_der) < 80):
        return "BRAZOS EN CRUZ"
    
    if brazo_izq_abajo and brazo_der_abajo:
        return "BRAZOS ABAJO"
    
    if brazo_der_arriba and not brazo_izq_arriba:
        return "BRAZO DERECHO"
    
    if brazo_izq_arriba and not brazo_der_arriba:
        return "BRAZO IZQUIERDO"

    return "BRAZOS RELAJADOS"


def ejecutar_accion_robot(posicion, acciones):
    if posicion == "BRAZO DERECHO":
        acciones.derecha()

    elif posicion == "BRAZO IZQUIERDO":
        acciones.izquierda()

    elif posicion == "BRAZOS RELAJADOS":
        acciones.adelante()

    elif posicion == "BRAZOS ABAJO":
        acciones.atras()

    elif posicion in ("BRAZOS EN CRUZ", "SIN BRAZOS"):
        acciones.quieto()
