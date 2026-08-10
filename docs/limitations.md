# Limitaciones

## Limitación observada del screen A0

A0 no aprende una dinámica temporal útil (Pearson log-ratio 0,046) y su weak-box
IoU ronda 0,50. La referencia release tampoco es causalmente comparable porque
sus tres secuencias de evaluación estaban expuestas durante entrenamiento. Hasta
completar un baseline matched, la tabla release no demuestra superioridad causal.

La descomposición localiza el fallo en la extensión temporal del foreground, pero
no demuestra todavía por qué esa extensión no se aprende. En particular, atribuir
el defecto a la weak-box rectangular es una hipótesis plausible y preregistrada,
no un resultado. También quedan por separar aliasing/representación, capacidad del
foreground head y conflicto entre pérdidas. A1 solo prueba la primera explicación.

Actualizado: 2026-08-10.

- No existe un claim SOTA ni un checkpoint final promovido.
- El screen causal-scale eAP está implementado pero no ejecutado completo. El único
  MiD real nuevo (`345.18`) procede de 128 validation tras un epoch de throughput y
  no es seleccionable.
- La comparación Garl event-only en los mismos 2.048 tokens está pendiente.
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
