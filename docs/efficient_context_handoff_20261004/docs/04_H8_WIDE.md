# 4. E2 — una prueba nueva: H8-WIDE

## Hipótesis
H16 modifica dos factores respecto a H8: aumenta las observaciones de 8 a 16 y
amplía la separación entre anchors de 350 a 750 ms. H8-WIDE conserva ocho
observaciones, distribuidas en el alcance de H16. Puede conservar evidencia útil
con menos trabajo por ventana, o perder detalles importantes. Es una prueba, no
una mejora asumida.

## Brazo y selección fijos
ID: `TPR-D1-H8WIDE-C160`.
Slots cronológicos del H16 padre: `[0,2,4,6,9,11,13,15]`.
Con los lags existentes: edades `[750,650,550,450,300,200,100,0]` ms.
No seleccionar por targets, confianza, pérdida o contenido del ROI.

En perfilado no ejecutar dieciséis observaciones para después descartar ocho.
Preparar y evaluar únicamente las seleccionadas. La unión raw sigue alcanzando
aproximadamente 1.050 ms, de modo que no se promete reducir I/O a la mitad.

Conservar D1, consultas, etiquetas, productores, PHASE17, normalizador histórico y
la GRU160 de H8/H16. Usar el objetivo original, con `lambda_cost=0.01`, no C0.
Verificar qué normalizador consumieron realmente los controles. No ajustarlo a
OLD_DEV ni sustituir un manifiesto de TRAIN por los índices de esta revisión.

Recalcular únicamente el canal de gap entre observaciones seleccionadas. Conservar
edades respecto al anchor, edades de disponibilidad y retrasos de disponibilidad.
No presentar observaciones antiguas como recientes. La máscara se obtiene del H16
padre: padding inicial, presente obligatorio y ningún cold start eliminado.
`reference/context_plan.py` define estos contratos; contrastarlo con el adaptador
real de contexto retrospectivo condicionado al ROI actual.

## Optimización
Seed7, folds 0/1/2: tres fits nuevos de 2.500 updates. CPU FP32, batch 128, AdamW con
wd 1e-3, warmup de 100 hasta 3e-4, cosine hasta 3e-5, clipping 1. Conservar los demás valores
realmente ejecutados de H16. Inicialización nueva por seed, no copia de pesos H8/H16.
Checkpoint completo cada 100 updates y progreso cada 25.

Congelar los tres endpoints antes de la primera evaluación OLD_DEV nueva.
Comparación primaria: WIDE−H8 de igual seed. Secundaria: WIDE−H16. H1 sirve como
referencia contextual. Misma población de 8.192 consultas y mismas masas, incluyendo
las secuencias difíciles. Aplicar los dos métodos de intervalos existentes.

## Regla de réplica, no de aceptación
Autorizar seeds 13/23 × tres folds cuando seed 7 cumpla simultáneamente:
- Delta puntual frente a H8 ≤ −1 MiD y mejora en al menos 6/9 secuencias.
- Salidas finitas 100 %; aumento de error de signo ≤0,5 puntos porcentuales.
- Incremento del bucket crucial ≤5 MiD y fuentes válidas.

No exigir significación estadística antes de gastar en replicación. Máximo de la
rama: 22.500 updates. Si la regla no se cumple, cerrar negativo o incierto sin probar
otros lags, anchos o backbones. Un bloqueo temporal del perfilado raw no prohíbe
clarificar la evidencia científica desde las cachés existentes.

Promediar pérdidas por consulta entre seeds, nunca TTC para crear un ensemble.
La independencia entre escenas sigue limitada a nueve secuencias. Separar seed 7
exploratoria de las dos nuevas. La comparación de precisión y coste no promueve
retroactivamente a WIDE ni elimina H8/H16.

Una posible no-inferioridad frente a H16 requiere su margen explícito: mantener
2 MiD como análisis ingenieril secundario y examinar el límite superior del IC.
Que el intervalo cruce cero no demuestra equivalencia.
