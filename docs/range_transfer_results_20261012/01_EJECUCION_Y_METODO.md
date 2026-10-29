# 01. Trabajo realizado y método

## Pregunta y alcance

La campaña intenta reducir la sobreestimación del TTC corto conservando precisión media/larga, negativos, cobertura y coste. Aislé dos factores en una cabeza temporal: la masa de las muestras en la pérdida y un término relativo adicional. No entrené nuevos encoders ni PAIR. Los seis fits usan los mismos productores event-only finales de V13.

## Identidades y preparación

- Evidencia histórica V12: `1a686e338a1145c6794bec0704bdae4cda95f06d`.
- Base auditada V13 y base del worktree V14: `da068ebdeb2d967ee10b2afa348e1317b8720ad0`.
- Rama: `scientific-recovery-v14-range-transfer`.
- ZIP de handoff: `E_JEPA_TTC_RANGE_TRANSFER_20261010.zip`, SHA256 `96a95c1ad8d63cbe5097e96b004c7f0d5f67e6a7707deb1ef6a7189e832ebb78`.

Leí el handoff y readmenow.txt, comprobé sus identidades, creé el worktree separado e integré el patch de referencia tras `git apply --check`. Inventarié procesos, recursos y finales disponibles. V14 no tomó la propiedad de la cola V13. Cuando V13 liberó OWNER.json tras terminar, registré su ausencia; no fabriqué un propietario ni reinicié la cola.

Verifiqué los padres, sus recibos finales, la procedencia P-only y los hashes de caché/normalizador H. El control V13 existente se entrenó en CUDA. Por eso entrené el control M0-L0 en CPU con la misma receta del factorial, en vez de tratar ambos endpoints como idénticos.

| Padre congelado | Updates | SHA256 del checkpoint |
| --- | --- | --- |
| E_A5_MATCHED | 30330 | `9f779a0376d6988d1e3f70ef566797f840d1fe4dbf204ce2f9ae1dad5322d512` |
| E_C2F_MATCHED | 30330 | `468f69aee68a0a5f8e2b536ba8c790ed45620251682aa0c70108625c426b30ec` |
| PAIR_E_MATCHED | 6840 | `3e503f7e1de5d7b4e460371f9bdfefb6dd7d59a280437b11258b21540994b883` |

## Separación de roles

| Rol | Uso | Filas | Grupos |
| --- | --- | --- | --- |
| P | Entrenamiento previo de productores y PAIR | 53890 | 24 |
| H | Normalizador y entrenamiento de cabezas | 18422 | 8 |
| V | Desarrollo interno: gates y selección | 16432 | 8 |

Comprobé grupos disjuntos y la correspondencia de tokens con manifiestos. H contiene features `[18422,8,17]`, timing `[18422,8,4]`, máscara `[18422,8]` y tres fases de expertos por consulta. Ocho slots es la capacidad de la cabeza: las máscaras y tiempos conservan las observaciones reales, sin densificar el historial.

El dominio admitido de H es `-10 < T < 0` o `0.1 < T <= 10`. La auditoría final encontró extremos aproximados −9,9993 y 9,9983 segundos. No eliminé targets para hacer viable el entrenamiento. La reutilización de la caché sensorial TRAIN40 no equivale a reutilizar sus productores: los pesos del factorial pertenecen a los padres matched entrenados en P.

## Matriz cerrada y factores

Entrené `M0/M1/M2 × L0/L1`, seed7, seis endpoints de 2.500 updates.

- **M0:** copié el vector nativo de masa sin renormalizarlo.
- **M1:** conservé la masa de cada grupo y repartí 0,5/0,3/0,1/0,1 entre crucial, media, larga y negativa presentes. Dentro de cada grupo y banda repartí la masa entre sus filas.
- **M2:** conservé la masa grupo×banda de M1. Busqué equilibrio entre (0,1;1], (1;2], (2;3] y entre (6;8], (8;10], acotando cada peso a [0,25;4] veces M1. El límite impide alcanzar igualdad perfecta cuando un subrango es escaso.

Las masas multiplican la contribución a la pérdida. El sampler sigue siendo uniforme con reemplazo, con los mismos draws por semilla. Los seis hashes de draws coinciden. Ninguna variante crea más episodios independientes mediante sus pesos.

**L0** conserva `simplex_t.model.training_loss`: fase, cuantiles y costes auxiliares. **L1** añade:

```text
z_i = (T_pred_i - T_target_i) / max(abs(T_target_i), 0.5 segundos)
L1 = L0 + 0.25 * mean_batch(N * mass_i * SmoothL1(z_i, 0; beta=1))
```

La predicción usa el mapping TTC nativo de `raw_location`. El target se reconstruye de su fase sin aplicarle el cap de salida. Conservé la política de emisión, la saturación y las regiones de gradiente plano; no introduje straight-through. La cabeza sigue siendo un corrector residual, no un router que seleccione el experto de menor coste auxiliar.

## Receta fija

CPU FP32, un thread numérico en el runner, batch128, GRU160 de dos capas, 17 features, salida residual, normalizador H congelado, AdamW (betas 0,9/0,999; eps 1e−8; weight decay 1e−3; foreach false), warmup100, pico LR 3e−4, final 3e−5, cosine y clipping de norma 1. La semilla fija inicialización y sampler. Todas las celdas parten de una inicialización nueva; ninguna parte del ganador de otra celda.

## Orden de ejecución y continuidad

1. N0: comprobación del paquete, estado vivo y recálculos públicos históricos.
2. N1: trazas históricas V12; controles H por hash y trazas matched. El informe 02 distingue la traza adicional terminada durante el factorial de los diagnósticos prefits.
3. N2: auditoría de masa, exposición y gradientes H sin updates.
4. Prueba sintética de continuidad: 10 updates frente a 5+restore+5.
5. Seis fits en el orden M0-L0, M1-L0, M2-L0, M0-L1, M1-L1, M2-L1. Guardado cada 100 updates y en pausa.
6. Congelación conjunta en SCREEN_ENDPOINTS.json, antes de inferir o puntuar esta matriz V.
7. Predicción V, métricas, bootstrap por grupos y selección con gates fijados.
8. Cierre NO_CANDIDATE, sin cuatro fits de réplica ni nueva evaluación de transferencia.

Cada checkpoint conserva pesos, AdamW, scheduler, cursor, RNG Python/NumPy/Torch, estado del sampler y contadores. El runner valida las fuentes y la admisión en fronteras durables. La publicación de informes de esta entrega no altera esos archivos científicos.

Fuentes: [admisión](evidence/ADMISSION.json), [roles](evidence/ROLE_AUDIT.json), [caché H final](evidence/H_FINAL_CACHE_AUDIT.json), [fuentes ejecutadas](evidence/SOURCE_BINDINGS.json), [endpoints](evidence/SCREEN_ENDPOINTS.json), [protocolo original](../range_transfer_handoff_20261010/docs/03_EXPERIMENT.md).
