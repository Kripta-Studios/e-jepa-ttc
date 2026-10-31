# Cierre del piloto JEPA: 31 de octubre de 2026

Estado: **cerrado; no se autoriza continuación automática de entrenamiento**.
Decisión aceptada por el usuario después de revisar el resultado. El piloto
terminó técnicamente, pero no superó su criterio principal de beneficio.
No se incorpora al PHASE-TTC publicado y no se inicia escalado ni rollout.

## Qué se ejecutó

Tres semillas independientes (7, 13, 21), cada una con destilación estática
(`B_static`), predicción de un paso (`C_one_step`) y futuro permutado entre
secuencias (`C_shuffle`). Cada brazo entrenó 1.000 actualizaciones, batch 16,
con el checkpoint final fijado antes de evaluar. Se utilizaron 512 muestras P
de 24 secuencias y 256 H de ocho secuencias distintas. Las semillas corrieron
simultáneamente en CPU, con dos hilos cada una. Los nueve entrenamientos y el
evaluador terminaron con código 0; el tiempo total fue 1.056,442 segundos.

El estudiante de eventos se entrenó frente a características espaciales fijas
de V-JEPA 2.1 ViT-B destilado. Las etiquetas TTC no entraron en sus pérdidas.
La representación congelada se evaluó después, retirando el predictor, con
ridge de alpha 1 ajustado exclusivamente en P. El objetivo principal fue el
cambio de log-altura de cajas anotadas a partir de diferencias temporales del
encoder. **No es una tasa normalizada por tiempo ni un error de TTC en segundos.**
La localización mediante cajas anotadas es información privilegiada del protocolo.

Este experimento es un filtro de representación sin supervisión TTC durante
el entrenamiento del estudiante. No ejecuta la matriz supervisada completa A–E
del planteamiento inicial ni una comparación final con el sistema publicado.
Tampoco reproduce el modelo completo de RoboJEPA, su profesor ViT-G, su
condicionamiento por acciones o su evaluación de planificación.

## Resultado principal

Las tablas se generan desde los JSON originales mediante
[analyze_jepa_pilot_closure.py](../../scripts/analyze_jepa_pilot_closure.py):
[tablas completas](tables.md), [deltas por secuencia](sequence_deltas.csv) y
[resumen regenerable](closure_analysis.json).

El criterio requería que C superase tanto B como el futuro permutado, con el
límite superior del intervalo bootstrap por secuencia por debajo de cero.
**Ninguna de las dos comparaciones principales cumple el criterio.** C mejora
la media frente a B, pero el intervalo cruza cero; frente al futuro permutado
su media es peor y el intervalo también cruza cero. No hay prueba de equivalencia
ni de imposibilidad general de beneficio: hay un filtro de continuación fallido.

La predicción del estado futuro del profesor mejora frente a ambos controles
y frente a persistir la salida presente del propio estudiante. Sin embargo,
la predicción del incremento entre estados empeora frente a ambos controles.
En B, el predictor residual permanece inicializado a cero, por lo que su
incremento es cero. Que C mejore el estado futuro no demuestra que aprenda
mejor el cambio físico necesario para TTC.

Una explicación posible es que el predictor compense errores de la
representación presente o aprenda aspectos del futuro que no ayudan a expansión.
**No se ha identificado causalmente ese mecanismo.**

## Revisión exploratoria posterior al experimento

Se conservaron las ocho secuencias y las tres semillas, sin seleccionar una
vista o checkpoint favorable. Se compararon, por secuencia, cambios de error
latente, error de expansión y acierto del signo. La mejora del estado futuro
aparece en las ocho secuencias en promedio frente a ambos controles, pero solo
dos mejoran también expansión frente a ambos: `CzyQ8HKP6X` y `JaGC4Q9pgU`.
Solo `JaGC4Q9pgU` mejora expansión frente a ambos controles en las tres semillas.
Ambas secuencias mejoran también el signo en promedio frente a ambos controles.

Las asociaciones descriptivas entre deltas de error futuro y expansión son
débiles en este conjunto (Pearson aproximadamente 0,21 frente a B y 0,27 frente
al futuro permutado; ocho puntos por comparación, semillas promediadas).
No se utilizan como pruebas de significación ni como evidencia de un mecanismo.
Las excepciones son post hoc: no constituyen un subgrupo validado ni cambian
el resultado principal. No se infirieron regímenes físicos a partir de los
identificadores opacos de las secuencias. No se entrenó ningún modelo nuevo.

## Alcance y límites

- H está separado de P dentro del piloto, pero son datos históricamente
  expuestos en el proyecto: no es confirmación independiente ni evidencia SOTA.
- El bootstrap remuestrea ocho secuencias y mantiene juntas las tres semillas;
  sus intervalos son condicionales a esas semillas, no describen toda la
  incertidumbre sobre futuras inicializaciones.
- No hay evaluación nueva de TTC en segundos, fusión final, robustez, adaptación
  supervisada completa o coste de inferencia de un predictor retenido.
- El rollout autorregresivo y el uso del predictor durante inferencia quedaron
  **sin ejecutar**, no refutados. La ausencia de evidencia de utilidad previa
  no justifica financiarlos automáticamente.
- Este resultado no identifica por sí solo la causa del fallo histórico de V8.

## Decisión y condiciones de reapertura

Se archiva esta receta como candidata de mejora inmediata. No se cambia el
modelo publicado, no se aumenta su escala y no se busca una configuración
favorable a partir de H. La revisión acotada no encontró un mecanismo validado
que justifique otra campaña.

Reabrir requiere una hipótesis concreta sustentada en evidencia nueva, un
criterio fijado antes de entrenar, controles estático y temporal permutado,
presupuesto acotado y una evaluación que distinga datos explorados de
confirmación independiente. Una secuencia favorable o que el rollout siga
pendiente no son razones suficientes. La prioridad vuelve al análisis de errores
del PHASE-TTC actual y a intervenciones directamente relacionadas con expansión,
signo y TTC en segundos; este cierre no lanza esa nueva investigación.

## Conservación y reproducción

Los originales permanecen en `/workspace/jepa_research_20261031/causal_pilot768/`.
El archivo `causal_pilot768_closed.tar.gz` se conserva tanto en el pod como en
`artifacts/jepa_council_20261031/` del workspace. Contiene 62 archivos, incluidos
los nueve checkpoints finales, código ejecutado, protocolo, registros,
resúmenes y predicciones. Los inputs, características intermedias y checkpoints
de recuperación siguen en el pod; el archivo compacto no contiene el dataset.

[CLOSURE.json](evidence/CLOSURE.json) contiene el SHA-256 del archivo y de cada
miembro. Se verificaron todos los miembros tras descargarlo. No había procesos
activos de este piloto al cerrarlo; no se detuvo ni modificó el controlador
ajeno de detección. La ausencia de procesos es una observación del cierre,
no un bloqueo técnico que impida a alguien lanzar manualmente los scripts.

Los JSON pequeños necesarios para regenerar el análisis están en
[evidence/](evidence/). Los archivos binarios voluminosos permanecen fuera de Git.
Desde la raíz del repositorio:

```bash
python -S scripts/analyze_jepa_pilot_closure.py --evidence docs/jepa_closure_20261031/evidence --output docs/jepa_closure_20261031
```

El análisis utiliza únicamente la biblioteca estándar. `-S` permite reproducirlo
sin cargar los paquetes del entorno de entrenamiento. Se usó también para evitar
un error de decodificación de un archivo `.pth` del entorno local existente;
ese entorno no se modificó. Regenerar el análisis no ejecuta entrenamiento.
