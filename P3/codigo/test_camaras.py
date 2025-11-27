#!/usr/bin/env python3
"""
Script de prueba para identificar los índices de las cámaras disponibles.
Ejecutar con: python test_camaras.py
"""

import cv2
import time

def listar_camaras_disponibles(max_camaras=5):
    """
    Prueba los primeros N índices de cámara para ver cuáles están disponibles.
    
    Args:
        max_camaras: Número máximo de índices a probar
    """
    print("=" * 60)
    print("BUSCANDO CÁMARAS DISPONIBLES")
    print("=" * 60)
    
    camaras_encontradas = []
    
    for idx in range(max_camaras):
        print(f"\nProbando índice {idx}...", end=" ")
        cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            # Intentar leer un frame para verificar que funciona
            ret, frame = cap.read()
            if ret:
                altura, ancho = frame.shape[:2]
                print(f"✓ ENCONTRADA ({ancho}x{altura})")
                camaras_encontradas.append({
                    'indice': idx,
                    'resolucion': (ancho, altura)
                })
            else:
                print("✗ No se pudo leer frame")
            cap.release()
        else:
            print("✗ No disponible")
    
    print("\n" + "=" * 60)
    print(f"RESUMEN: {len(camaras_encontradas)} cámara(s) encontrada(s)")
    print("=" * 60)
    
    if camaras_encontradas:
        for cam in camaras_encontradas:
            print(f"  - Índice {cam['indice']}: {cam['resolucion'][0]}x{cam['resolucion'][1]}")
    else:
        print("  No se encontraron cámaras disponibles")
    
    return camaras_encontradas


def test_camara_especifica(indice):
    """
    Muestra el feed de una cámara específica.
    
    Args:
        indice: Índice de la cámara a probar
    """
    print(f"\n{'=' * 60}")
    print(f"PROBANDO CÁMARA {indice}")
    print(f"{'=' * 60}")
    print("Presiona 'q' para salir y probar otra cámara")
    print("Presiona 's' para tomar una captura de pantalla")
    
    cap = cv2.VideoCapture(indice)
    
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {indice}")
        return False
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    capturas = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: No se pudo leer el frame")
                break
            
            # Añadir información al frame
            cv2.putText(
                frame,
                f"Camara {indice} - Presiona 'q' para salir, 's' para captura",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 255, 10),
                2
            )
            
            cv2.imshow(f"Test - Camara {indice}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"captura_camara_{indice}_{capturas}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Captura guardada: {filename}")
                capturas += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Cámara {indice} cerrada")
    
    return True


def test_ambas_camaras(indices):
    """
    Muestra ambas cámaras simultáneamente.
    
    Args:
        indices: Lista con los índices de las cámaras [webcam, smartphone]
    """
    if len(indices) < 2:
        print("ERROR: Se necesitan al menos 2 cámaras")
        return
    
    print(f"\n{'=' * 60}")
    print(f"PROBANDO AMBAS CÁMARAS SIMULTÁNEAMENTE")
    print(f"{'=' * 60}")
    print(f"Webcam (telecontrol): índice {indices[0]}")
    print(f"Smartphone (detección): índice {indices[1]}")
    print("Presiona 'q' para salir")
    
    cap_webcam = cv2.VideoCapture(indices[0])
    cap_smartphone = cv2.VideoCapture(indices[1])
    
    if not cap_webcam.isOpened() or not cap_smartphone.isOpened():
        print("ERROR: No se pudieron abrir ambas cámaras")
        cap_webcam.release()
        cap_smartphone.release()
        return
    
    # Configurar resolución
    for cap in [cap_webcam, cap_smartphone]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret1, frame_webcam = cap_webcam.read()
            ret2, frame_smartphone = cap_smartphone.read()
            
            if not ret1 or not ret2:
                print("ERROR: No se pudo leer alguno de los frames")
                break
            
            # Añadir etiquetas
            cv2.putText(
                frame_webcam,
                f"WEBCAM TELECONTROL (idx {indices[0]})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                frame_smartphone,
                f"SMARTPHONE DETECCION (idx {indices[1]})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Webcam (Telecontrol)", frame_webcam)
            cv2.imshow("Smartphone (Deteccion)", frame_smartphone)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap_webcam.release()
        cap_smartphone.release()
        cv2.destroyAllWindows()
        print("Cámaras cerradas")


def main():
    """Menú principal interactivo"""
    print("\n" + "=" * 60)
    print("TEST DE CÁMARAS - ROBOBO PROJECT")
    print("=" * 60)
    
    # Listar cámaras disponibles
    camaras = listar_camaras_disponibles()
    
    if not camaras:
        print("\nNo se encontraron cámaras. Verifica las conexiones.")
        return
    
    while True:
        print("\n" + "-" * 60)
        print("OPCIONES:")
        print("-" * 60)
        print("1. Probar una cámara específica")
        print("2. Probar ambas cámaras simultáneamente")
        print("3. Volver a buscar cámaras")
        print("4. Salir")
        print("-" * 60)
        
        opcion = input("Selecciona una opción (1-4): ").strip()
        
        if opcion == '1':
            indice = input(f"Índice de la cámara a probar (0-{len(camaras)-1}): ").strip()
            try:
                indice = int(indice)
                test_camara_especifica(indice)
            except ValueError:
                print("ERROR: Índice inválido")
        
        elif opcion == '2':
            if len(camaras) < 2:
                print("ERROR: Se necesitan al menos 2 cámaras")
            else:
                idx1 = input("Índice de la WEBCAM (telecontrol): ").strip()
                idx2 = input("Índice del SMARTPHONE (detección): ").strip()
                try:
                    test_ambas_camaras([int(idx1), int(idx2)])
                except ValueError:
                    print("ERROR: Índices inválidos")
        
        elif opcion == '3':
            camaras = listar_camaras_disponibles()
        
        elif opcion == '4':
            print("\n¡Hasta luego!")
            break
        
        else:
            print("ERROR: Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    finally:
        cv2.destroyAllWindows()
