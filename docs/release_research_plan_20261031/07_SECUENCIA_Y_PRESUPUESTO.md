# Secuencia de investigación y presupuesto acotado

## Orden de ejecución

| Etapa | Trabajo | Condición para pasar |
|---|---|---|
| P0 | Congelar R0, resolver artefactos, reproducir baseline y exposición | Identidad y replay coherentes; población declarada |
| P1 | Incorporar resultado completo C-TIME y diagnóstico espacial V0 | Cambio atribuido y cobertura conservada; no basta simetría |
| P2 | O0: utilidad de corrección y soporte; D0–D3: observador/control | Evidencia de un error corregible en el release |
| P3 | Un brazo aprendido: O2 **o** D4 | Señal frente a control de capacidad/updates y sin daño material |
| P4 | Confirmación del candidato y conservación nativa | Criterios fijados, adquisición independiente o alcance limitado declarado |
| P5 | S0–S3 o V2/V3, según fallo residual | Requisitos de dato satisfechos y nueva hipótesis concreta |
| P6 | Combinación de dos brazos admitidos | Interacción favorable frente a cada brazo por separado |

La propuesta prioriza decisiones, no una fecha de lanzamiento. No se estima duración total de GPU sin medir lectura RAW, cachés, reconstrucción OOF y hardware. Las adquisiciones/calibración pueden ser el camino crítico.

## Presupuesto inicial propuesto

Estas cifras son límites de screening, no evidencia de convergencia ni autorización de gasto en servicios externos. Registrar cualquier cambio de presupuesto antes de evaluar resultados.

| Trabajo | Techo inicial | Qué se cuenta |
|---|---|---|
| Baseline/cadencia | Cero updates; una extracción por identidad de entrada | Productores, historia y replay; reutilización solo sellada |
| O0/V0 | Una matriz fija de diagnósticos | Todas las consultas, sin búsqueda de umbrales por test |
| Gate O1/O2 | Máximo cuatro brazos aprendidos, 1.000 updates, una semilla en screen | Control reajustado y controles físicos incluidos |
| Contexto D4 | Máximo cuatro brazos, 1.000 updates, una semilla en screen | Contexto libre, original reajustado y permutación |
| Sensor S2 | Máximo cuatro brazos, 1.000 updates, una semilla tras S0/S1 | Preparación fotométrica y reconstrucción superior se cargan aparte |
| Confirmación de una familia | Tres semillas del bloque realmente modificado | Mismos datos, receta, endpoint y ancestros admitidos |

No ejecutar las tres familias aprendidas a la vez. Una familia fallida no habilita automáticamente probar todas las demás. El diagnóstico residual determina el siguiente brazo.

Una semilla sirve para detectar señal o errores técnicos; no acredita robustez. Si 1.000 updates no permiten ni aprender el control positivo, declarar ensayo insuficiente. No aumentar el presupuesto solo al brazo físico que pierde. Antes de medir confirmación, fijar un endpoint común con curvas de entrenamiento/desarrollo y documentar la decisión.

## Presupuesto de tiempo y memoria

Medir 100 pasos de perfilado o un bloque de extracción representativo antes del screen. Estimar por separado inicialización, lectura, preparación, forward/backward, guardado y generación de features. Registrar velocidad sostenida, memoria máxima, almacenamiento proyectado y número de fits auxiliares.

Si se modifica un productor, el coste de su experimento no es solo un fine-tune: incluye sus rotaciones y descendientes cuando se reclame OOF. Si ese coste excede el techo disponible, mantener un piloto exploratorio de desarrollo y explicitar que falta confirmación. No renombrar cachés in-sample como OOF para abaratarlo.

Mantener un solo proceso pesado de investigación por recurso compartido salvo capacidad comprobada. Este plan no duplica `population2608_v1`, no altera sus fuentes selladas ni reutiliza un PID sin verificar proceso y comando actuales.

## Reglas de parada

1. **Fallo de contrato:** detener la comparación y corregir identidad, causalidad o datos antes de interpretar métricas.
2. **Sin señal de mecanismo:** si la variable propuesta no discrimina el fallo objetivo, no entrenar una cabeza mayor sobre ella.
3. **Mecanismo correcto, TTC peor:** conservar el resultado; comprobar target, sesgo y distribución antes de combinar módulos.
4. **Control igual de bueno:** atribuir el beneficio a reajuste/capacidad/calibración, no a la física propuesta.
5. **Beneficio solo en un dominio:** mantener candidato especializado con alcance explícito; no sustituir R0 global.
6. **Confirmación imprecisa:** declarar resultado inconcluso; no transformar ausencia de significación en equivalencia.
7. **Coste excesivo:** conservar una mejora científica si existe, pero no promoverla como producción sin objetivo de coste satisfecho.

## Combinación permitida

Si C-TIME y C-OBS aportan por separado, comparar R0, TIME, OBS y TIME+OBS en un factorial 2×2 con iguales datos. El gate puede requerir recalibración por cambio de distribución; igualar ese reajuste en los controles. Hacer lo mismo con CLOCK solo después de comprobar que no duplica la misma corrección temporal.

Si un auxiliar cambia el productor, la combinación posterior vuelve a generar mediciones e incertidumbre. No heredar una calibración anterior porque tenga el mismo nombre de campo. Confirmar de nuevo precisión y coste de la cadena completa.
