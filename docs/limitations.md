# Limitaciones

## Limitación observada del screen A0

A0 no aprende una dinámica temporal útil (Pearson log-ratio 0,046) y su weak-box
IoU ronda 0,50. La referencia release tampoco es causalmente comparable porque
sus tres secuencias de evaluación estaban expuestas durante entrenamiento. Hasta
completar un baseline matched, la tabla release no demuestra superioridad causal.

La descomposición localiza el fallo en la extensión temporal del foreground. A1 ya
probó la hipótesis weak-box: retirarla mejora altura y MiD, pero anchura/centros y
dinámica siguen débiles. Por tanto el rectángulo era ruido parcial, no explicación
suficiente. Quedan por separar representación/aliasing event-native, capacidad del
foreground head y coherencia temporal.

La auditoría de actividad usa momentos de `abs(events)` como heurística, no como
segmentación ni demostración causal. Que la actividad sea difusa y el decoder use
`amax` axial justifica probar una cabeza 2-D, pero solo el experimento controlado
puede confirmar que el colapso axial sea la causa.

El primer intento full-resolution no es evidencia negativa ni positiva: el selector
aceptó una métrica macro que omitía una secuencia no finita. Está invalidado y
preservado. Solo la repetición desde cero con cobertura 3/3 puede evaluar el brazo.

La repetición válida confirma un resultado negativo, no la causa completa: una
cabeza 2-D superficial sobre inputs crudos empeora A1. Esto no demuestra que toda
representación 2-D o pretraining falle; motivó el control profundo A1-DF posterior.

A1-DF sí alimenta foreground desde features profundas y mejora correlaciones, pero
no MiD/failure. La oscilación por época es compatible con interferencia entre
objetivos, aunque no la demuestra causalmente. El siguiente pair-ratio es una
ablation de esa hipótesis, no una reparación garantizada ni un gate revisado.
El peso `5.0` se normalizó con train, pero sigue siendo una única elección de diseño;
un resultado negativo no descarta todos los pesos y no autoriza un sweep post-hoc.

Actualizado: 2026-08-10.

- No existe un claim SOTA ni un checkpoint final promovido.
- A0 sí se ejecutó completo y fue negativo; el antiguo MiD parcial `345.18` queda
  supersedido por MiD macro `382.19` del checkpoint seleccionable.
- La referencia Garl release en los mismos 2.048 tokens está completa, pero la
  comparación matched desde cero ya está completa con una sola seed.
- Garl matched (`203.63` MiD macro, cero failures) es una referencia controlada de
  arquitectura, no evidencia multisemilla ni resultado oficial. Su salida está
  colapsada al signo positivo: las 335 filas negative/receding reciben TTC positivo
  y MiD `437.60`. No debe promoverse ignorando esta limitación.
- A1 está ejecutado y es un resultado negativo parcial: mejora MiD macro a
  `346.83`, pero queda `143.20` detrás de Garl matched, con `9.96%` failures.
  Supervisar bbox geometry no produjo medidas buenas de anchura/centros ni una
  dinámica temporal suficiente. La bbox sigue siendo supervisión oracle
  training-only y el crop continúa siendo bbox oracle.
- El screen depende de ROI con cajas GT y weak-box supervision; no es bbox-free y
  sus rectángulos no son máscaras de segmentación.
- Resume atómico pasa su prueba end-to-end, pero el run real representativo sigue
  pendiente.
- Causal Scale v5 aprende foreground en datos sintéticos y alcanza IoU `.8640`, pero
  el test held-out falló: Pearson `.92135 < .95` y translation leakage
  `.02749 > .02`. Seed 303 está consumida y no hay evidencia real.
- V6 corrige translation leakage (`.00462`) e IoU (`.89323`) en validation nueva,
  pero Pearson `.92042` sigue bajo `.95`; test 603 permanece sellado.
- V7 superó validation 502 pero falló held-out 603 en Pearson (`.92014 < .95`);
  seed 603 está consumida y no demuestra transferencia real o superioridad externa.
- V8 CVaR mejora Pearson macro a `.94621`, pero sigue bajo `.95`; test 901/902/903
  permanece sin abrir.
- El gate de rotación v5 usa un cuadrado controlado; no demuestra invariancia a la
  rotación de objetos generales.
- La propagación de incertidumbre v5 es una aproximación local y no es fiable cerca
  de la singularidad de TTC sin respetar `known_mask`.
- El único smoke raw high-resolution usa 16/16 muestras y obtiene MiD macro
  `1868,3186`; solo valida integración.
- Falta pretraining JEPA denso compatible. El candidato actual parte de random.
- El rango predictor real ≈1,10 demuestra capacidad efectiva baja, pero no prueba
  qué factor semántico domina porque faltan embeddings/probes reales compatibles.
- El benchmark de shortcut es sintético: demuestra un modo de fallo del objetivo,
  no que eAP sufra exactamente el mismo shortcut.
- VISReg y R²-lite no corrigieron el caso lento bajo el gate predeclarado. R² queda
  rechazado para producción por beneficio insuficiente y complejidad.
- El residual temporal mejoró el shortcut constante, pero perjudicó gravemente el
  control frame-varying; solo es válido como canal adicional, nunca como sustituto
  global del embedding de nivel.
- INTACT no puede evaluarse sin acciones expertas y no corresponde al objetivo TTC
  puramente perceptivo actual.
- RGB-E no está implementado en el trainer high-resolution.
- Faltan seis secuencias eAP y la evaluación oficial eAP/CodaBench.
- EvTTC Tabla VI carece de un manifest de inputs label-free real y congelado.
- Dense Patch gana un split histórico pero pierde grouped CV 5x3 frente a A0.
- AttnRes, KDA/Object-KDA y bbox-ROI no superan sus gates.
- La geometría bbox usa oracles o más contexto; STRTTC tiene cobertura incompleta.
- No existe todavía una estimación causal bbox-free de expansión/FoE que mejore A0.
- La compensación traslacional requiere profundidad; usar distancia GT es oracle.
- Family-OOD degrada mucho frente a validation.
- CARLA SSL/TTC sintético empeoró la transferencia; el dataset fue retirado y solo
  se conservan métricas negativas compactas.
- La caché Garl full se estima en ~455 GiB y no es viable. El trainer raw evita el
  disco, pero el I/O HDF5/voxelización puede limitar el throughput.
- Faltan tres seeds full, robustez/calibración reales, latencia end-to-end, ONNX y
  demo del checkpoint final.
- El sistema no está validado para control de seguridad.

Estas limitaciones impiden afirmar superioridad sobre Garl-TTC, EvTTC geométrico o
cualquier benchmark externo.
