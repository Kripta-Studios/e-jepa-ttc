# Integración archivo por archivo

Nombres nuevos propuestos; comprobar antes las rutas reales y adaptar sin duplicar lógica.

| Archivo/módulo | Trabajo |
|---|---|
| `src/e_jepa_ttc/models/causal_scale_ttc.py` | REUTILIZAR modalidad RGB, geometría y transporte. Mantener pruebas de regresión event. No copiar otro encoder grande. |
| `src/e_jepa_ttc/simplex_t/model.py` | Reutilizar GRU para R_PHASE17 mediante wrapper/identidad distinta. No hacer coincidir normalizadores o pesos por ancho. |
| `src/e_jepa_ttc/rgb_port/contracts.py` | Versionar modalidad, frames, tiempos, ROI, outputs y roles. Tomar reference como guía y probar los contratos reales. |
| `src/e_jepa_ttc/rgb_port/data.py` | Índice label-free de frames, lectura/decodificación RGB, uint8→[0,1], crop común y target batch separado. |
| `src/e_jepa_ttc/rgb_port/history.py` | Tripletas reales, hasta8 observaciones dentro650ms, deltas enteros, edad de disponibilidad y padding/máscaras. |
| `src/e_jepa_ttc/rgb_port/features.py` | R_PHASE17, diagnostics de expertos RGB y PAIR133. Sin datos GT ni diagnósticos event fingidos. |
| `src/e_jepa_ttc/rgb_port/normalization.py` | TRAIN/H-only sobre IDs únicos, hash de esquema, productor, partición y modalidad. |
| `src/e_jepa_ttc/rgb_port/fusion.py` | Dos streams con clocks propios, F_TRUE/F_ZERO, pérdida consistente, wrapper fallback exacto. |
| `operational/rgb_port/train_producers.py` | Adaptar motor causal-scale con inputs RGB/targets explícitos. Configs, gradientes, optimizer/RNG/cursor y freeze. |
| `operational/rgb_port/train_heads.py` | PAIR y seis heads con sus epochs/updates registrados; reutilizar durabilidad existente. |
| `operational/rgb_port/run.py` | Una cola DAG P0–P4, estado por fase, bloqueo por rama y resume sin duplicación. |
| `operational/rgb_port/evaluate.py` | Score reproducible, cohortes completas, métricas/buckets, contrasts y evaluación de transferencia separada. |
| `operational/rgb_port/profile.py` | Cabeza sola, productores, RGB encode/decode, ruta completa; una cabeza H8 canónica y tres cabezas por separado. |
| `operational/rgb_port/package.py` | Weights, preds, normalizers, source hashes, lineage y scripts de reproducción. |
| `configs/rgb_port/*.json` | Configs totalmente resueltas antes del primer update, sin defaults ocultos. |
| `tests/test_rgb_port_*.py` | Datos, forward/backward, causalidad, no fugas, fallback, paridad event, resume y metrics. |

## Pruebas necesarias además de los contratos de referencia

1. Forward/backward real del CausalScaleTTC RGB; comprobar que cambian pesos de encoder, no sólo la salida.
2. Derivación real de R_PAIR y R_PHASE17 sin acceder a labels.
3. Dos versiones de datos con TTC modificado tienen idéntico índice de contexto.
4. Alterar cualquier frame posterior al cutoff no cambia la salida ni el cache-key actual.
5. RGB normalizado ImageNet no entra al `_sensor_support` [0,1]; color RGB/BGR probado con fixture.
6. Diferente crop/resize o checkpoint fuerza cache miss; misma key sólo reutiliza salidas idénticas.
7. GRU usa máscara sin update en padding; RGB no duplica IDs para alcanzar ocho observaciones.
8. F_ZERO no recibe nada calculado de RGB salvo disponibilidad/tiempos controlados; F_TRUE responde a cambios RGB cuando entrenado.
9. Fallback sin RGB coincide con E_CTX e invoca cero encoders RGB.
10. Continuo versus checkpoint/resume igual en CPU y dentro de tolerancia predeclarada GPU, incluidos scheduler/RNG/optimizer y sampler.
11. Split/teacher/checkpoint de P no intersecta H o V; ajustar normalización con V lanza error.
12. La serialización nueva usa UTF-8 y LF explícitos; no alterar evidencias antiguas para hacerlas encajar.

## Caché que sí puede ahorrar en RGB

Dentro de una misma ROI de consulta, varias tripletas comparten frames RGB EXACTOS. El encoder por frame puede ejecutarse una sola vez por (frame ID, ROI, resize, normalización, checkpoint, precisión). Reutilizar los mapas densos y logits ANTES del suavizado temporal, no tokens geométricos ya dependientes de una tripleta distinta. Rehacer el suavizado, matching y readout con sus deltas reales.

Primero validar `forward normal` frente a `forward desde endpoints cacheados` con pesos congelados. En entrenamiento end-to-end no cachear features de un encoder que cambia cada update como si fueran inputs constantes.

No afirmar que esto elimina el coste de A5+C2F; ambos encoders conservan pesos diferentes. No reutilizar automáticamente entre queries con ROIs distintas ni hacer ROI pooling sobre un encoder de frame completo diciendo que es equivalente.
