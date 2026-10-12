# Investigación posterior: prioridades y experimentos, no ejecución automática

## Prioridad 1: no duplicar V15

V14 cerró la hipótesis M0/M1/M2 × L0/L1 con NO_CANDIDATE. V15 propone investigar muestreo de importancia y saturación de readout A5; este handoff no acredita su ejecución. Consultar rama, recibos y propietario reales. Si ya está activo, mantenerlo intacto; si no se inició, la orden V15 existente sigue siendo una propuesta separada, no algo realizado por este paquete.

Los resultados event-only V13 son inferiores a H8 V12 en MAE externo, pero no se entrenaron con los mismos grupos ni igual contexto/objetivo. Un refit de los productores finales sobre TRAIN40 es razonable para un modelo de entrega, no una prueba causal de que una arquitectura sea mejor. No cargar TRAIN40 para fingir un holdout P/H/V limpio.

Si V15 modifica sólo la cabeza, puede estudiarse un fallback nuevo con el mismo PHASE17, como política explícita de sistema. Si modifica el readout A5, reconstruir toda feature dependiente, normalizador y fusión correspondiente. No sustituir padres silenciosamente en F_TRUE.

## Prioridad 2: contrastar y estabilizar la contribución event a una ruta RGB fuerte

Hipótesis de trabajo, NO demostrada: la fusión actual podría beneficiarse de preservar explícitamente una estimación RGB refinada razonable y aprender sólo el complemento event, en lugar de volver a aprender dos GRUs y corregir la mediana de expertos event.

La evidencia motivadora es concreta: F_TRUE sólo gana mean_MiD en 16/32 secuencias Dev32 frente a R_CTX; su MAE es peor en estimación puntual, mientras en FCWD RGB disponible sí mejora a R_CTX. Eso no autoriza descartar eventos. Hace interesante una fusión que aproveche información cuando aporta y no agrande errores de discrepancia.

### Un único estudio posterior propuesto

Dos brazos, 2.500 updates cada uno, seed7, y comparadores congelados R_CTX/F_TRUE/E_CTX:

- `R_ANCHOR_EVENT`: usa estados finales de R_CTX y E_CTX congelados, sus fases y tiempos reales para aprender una corrección a `raw_location_R`.
- `R_ANCHOR_ZERO`: misma capacidad, mismos tiempos y receta, pero anula toda información predictiva event que no sea necesaria para el fallback. Incluye anular fase event, discrepancias event y su estado oculto, no sólo un vector parcial.

Forma conceptual:

```
raw_location = raw_location_R + 0.03 * MLP(h_R, h_E, fases, tiempos)
TTC = conversión nativa(raw_location)
```

Última capa inicializada a cero: la predicción inicial coincide exactamente con R_CTX en la misma precisión. No imponer un nuevo cap pequeño arbitrario al residual: precisamente se está investigando en V15 la restricción de rango. No convertir el signo a positivo, elegir siempre el TTC más corto o promediar TTC de signos opuestos. Los intervalos quedan como diagnósticos sin calibración asumida.

Las dos rutas emplean igual objetivo nativo de fase/cuantiles sin costes auxiliares de expertos excluidos, masa H y sampler fijos. Si se reutilizan estados de cabezas ya entrenadas en H, se declara como nueva etapa supervisada sobre H; esos estados no son OOF ni un teacher independiente. V queda excluido de optimizador/normalizador. El comparador nativo se conserva con su coste total de entrenamiento; no atribuir toda mejora a una única conexión residual sin más controles.

No hay garantía de que el evento complementario funcione. Un gate suave tampoco es por sí mismo fiabilidad calibrada. Una versión con calidad geométrica necesitaría datos TRAIN de perturbaciones coherentes y control de calidad; no añadir NCC de FCWD como feature de una cabeza entrenada sin ella.

Antes de ejecutar esta prueba hay que fijar el contrato concreto de estados ocultos, paridad del readout, controles, incertidumbre por grupos y umbrales prácticos. La receta no se cambia después de consultar V. Requiere una autorización nueva; este paquete tiene cero updates y no ejecuta los 5.000 propuestos.

## Prioridad 3: diversidad, representación y entrenamiento completo

Si el fracaso está antes de PHASE17 y V15 no lo corrige, la próxima intervención debe cambiar evidencia o representación, no otra ponderación sobre la misma cabeza:
- Medir en P/H episodios de aproximación, contexto/velocidad, tamaño angular, oclusión, eventos escasos y errores de etiqueta/anchor admisibles. No inferir dominio sólo por nombres de secuencia.
- Recoger/usar grupos permitidos nuevos y anotaciones confiables; más ventanas de los mismos episodios no equivalen a más escenas.
- Probar campos de correspondencia antes del promedio frente al resumen actual usando un mismo productor y control. La antigua variante LATENT global negativa no descarta campos espaciales.
- Si se aumenta rango/capacidad del encoder, rehacer dependencias downstream y comparar con igual información, no juntar pesos de sistemas distintos sin normalización/admisión.

## Ruta de publicación y réplica final

Sólo después de seleccionar una receta con evidencia interna/transferencia declarada:
1. Congelar arquitectura, datos permitidos, interpretación TTC, tipos de input y reglas de fallos.
2. Replicar el sistema completo (productores, PAIR y heads) con seeds13/23; no sólo última capa. Presentar resultados por seed y coste.
3. Planificar refit final TRAIN40 con cross-fitting para heads/correctores o un esquema de validación TRAIN interno explícito. No usar predicciones in-sample como si fueran OOF.
4. Congelar antes de producir test12; validar tokens/formato y registrar una única comparación oficial planificada. Enviar sólo tras autorización explícita de submission. El score privado no entra en una búsqueda iterativa.
5. Reportar por modalidad y presupuesto: event-only, RGB (y su dependencia real de ROI/registro), fusión, inputs nativos y contexto extra. Comparar Garl full/visual/event y alternativas TTC sólo cuando sus protocolos sean comparables.

No es defendible prometer ganar todas las métricas/escenas/latencias con un único escalar. La meta es dominar un conjunto de criterios fijados y documentar compromisos. El avance actual ya es real, pero el gap de Garl full en FCWD y la falta de confirmación permanecen.
