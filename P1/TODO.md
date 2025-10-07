
# cosas inmediatas que se me van ocurriendo
- [ ] Añadir sensores IR
- [ ] Desglosar todo lo relacionado con robobo en un archivo aparte, que haga de API


---

# enunciado de la practica

- [ ] Definición del problema e implementación de los elementos básicos: **(hasta 4 puntos)**:
    - [ ] Implementación del entorno de trabajo de RobobSim adaptado a Gymnasium.
    - [ ] Definición y complejidad de los espacios de estados, de acciones y función de recompensa. Cuanto mayor sea la complejidad y definición de los mismos, mayor será la puntuación. Ejemplo: la utilización de espacios continuos frente a espacios discretos es más compleja, pero se valorará más.
    - [ ] Se valorará el refinamiento de la función de recompensa, teniendo no sólo en cuenta las recompensas, sino también las posibles penalizaciones por incurrir en estados no deseados.
- [ ] Implementación del algoritmo de aprendizaje y calidad de la solución propuesta **(hasta 4 puntos)**:
    - [ ] Rapidez con la que se obtiene una solución aceptable (en pasos de tiempo)
    - [ ] Calidad de la solución obtenida. Es decir, consistencia con la que se aproxima al objetivo y rapidez con la que se aproxima.
- [ ] Representación de la información en un formato distinto a tensorboard **(hasta 2 puntos)**:
    - [ ] Representación de las métricas más relevantes dadas por el StableBaselines3 utilizando una librería diferente a tensorboard.
    - [ ] Representación en un plano 2D de las diferentes posiciones recorridas por el Robobo (y las posiciones que ha tomado el cilindro de ser el caso)
