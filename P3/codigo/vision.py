import numpy as np

def detectar_posicion_brazos(keypoints):
    if len(keypoints) < 11:
        return "SIN BRAZOS"

    keypoints = np.array(keypoints)
    hombro_izq, hombro_der = keypoints[5], keypoints[6]
    codo_izq, codo_der = keypoints[7], keypoints[8]
    muñeca_izq, muñeca_der = keypoints[9], keypoints[10]

    # Verificación de detección
    brazos_detectados = {
        'izq': muñeca_izq[0] > 0 and muñeca_izq[1] > 0,
        'der': muñeca_der[0] > 0 and muñeca_der[1] > 0
    }
    if not brazos_detectados['izq'] and not brazos_detectados['der']:
        return "SIN BRAZOS"

    # Alturas relativas
    altura_hombro_izq, altura_hombro_der = hombro_izq[1], hombro_der[1]
    altura_muñeca_izq, altura_muñeca_der = muñeca_izq[1], muñeca_der[1]
    x_muñeca_izq, x_muñeca_der = muñeca_izq[0], muñeca_der[0]

    # Distancia entre hombros para normalizar
    ancho_hombros = abs(hombro_der[0] - hombro_izq[0])
    if ancho_hombros == 0:
        ancho_hombros = 1

    distancia_muñecas = abs(x_muñeca_der - x_muñeca_izq)

    # Umbrales relativos
    brazo_izq_arriba = altura_muñeca_izq < altura_hombro_izq - 0.5 * ancho_hombros
    brazo_der_arriba = altura_muñeca_der < altura_hombro_der - 0.5 * ancho_hombros
    brazo_izq_abajo = altura_muñeca_izq > altura_hombro_izq + 1.0 * ancho_hombros
    brazo_der_abajo = altura_muñeca_der > altura_hombro_der + 1.0 * ancho_hombros

    # Nuevos gestos
    if brazo_izq_arriba and brazo_der_arriba and distancia_muñecas < 0.5 * ancho_hombros:
        return "MANOS JUNTAS ARRIBA"
    if brazo_izq_arriba and brazo_der_arriba and distancia_muñecas > 1.5 * ancho_hombros:
        return "BRAZOS EN CRUZ"
    if brazo_izq_abajo and brazo_der_abajo:
        return "BRAZOS ABAJO"
    if brazo_der_arriba and not brazo_izq_arriba:
        return "BRAZO DERECHO"
    if brazo_izq_arriba and not brazo_der_arriba:
        return "BRAZO IZQUIERDO"
    if distancia_muñecas < 0.3 * ancho_hombros and abs(altura_muñeca_izq - altura_muñeca_der) < 0.2 * ancho_hombros:
        return "MANOS JUNTAS PECHO"
    
    return "BRAZOS RELAJADOS"

