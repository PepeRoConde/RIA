# Practica 2
## Practica 2.1

En el escenario `CYLINDER`

    python3 -m venv venvRomeroFerreiro
    source venvRomeroFerreiro/bin/activate
    pip install -r requirements.txt
    
    python P2/codigo/entrenamiento.py
    python P2/codigo/test.py

## Practica 2.2 

En el escenario `AVOID THE BLOCK` cambiar (descomentar y comentar) el `config.yaml`, pasar de `posicion_inicial: null` a `posicion_inicial: {'x': -1000.0, 'y': 39, 'z': -400.0 }`.

    python P2/codigo/entrenamiento.py
    python P2/codigo/test.py

## Practica 2.2 

En el escenario `AVOID THE BLOCK` volver a cambiar (descomentar y comentar) el `config.yaml`, pasar de `posicion_inicial: {'x': -1000.0, 'y': 39, 'z': -400.0 }` a `posicion_inicial: null`.

    python P2/codigo/entrenamiento.py
    python P2/codigo/test.py
