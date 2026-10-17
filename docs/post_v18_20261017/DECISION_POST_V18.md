# Decisión posterior a V18

Conservamos V18 cerrado con `NO_CANDIDATE` y su entrega verificada. Ninguna receta cumplió conjuntamente los criterios de promoción. Los resultados incluyen mejoras parciales que justifican decisiones posteriores con un protocolo distinto.

La referencia documental es `a811cc3a39334532af457034bab5457127335bbd`; el ZIP contiene el código `c3f14408babda11f9f9d50ab1e4158992e6f9c19`. Esta revisión añade documentos y análisis CPU de tablas guardadas. Autoriza cero optimizer updates y ninguna inferencia nueva de comparación, submission, push o merge.

## Decisiones y entregables

| Identidad | Decisión | Evidencia y límite |
|---|---|---|
| PHASE_ONLY_EVENT_S7 V16 | Mantener el candidato congelado para evaluación oficial | Usa RGB y eventos, con historial causal. El ZIP y sus valores pasan la comprobación local; falta confirmar el historial de envíos de la cuenta y la admisibilidad. |
| FUSION-C1 V18 | Conservar como alternativa para comparación congelada | Mejora interna parcial en V; no pasa el criterio crucial histórico. La diferencia frente a V16 mantiene un intervalo que incluye cero. |
| A1-FULL V18 | Conservar como evidencia de mejora positiva y regresión negativa | Los resultados H no se comparan directamente con los scores V de la fusión. El nuevo desglose usa exclusivamente P. |

El [registro de candidatos](CANDIDATE_REGISTRY.json) distingue modalidades, semillas, genealogía y hashes. Las medias de métricas de tres semillas no son un ensemble. No seleccionamos la semilla 13 por su resultado.

## Evaluación oficial

El [estado oficial](OFFICIAL_EVALUATION_STATUS.json) registra la verificación actual del ZIP, CRC, formato, 6.762 tokens y cada valor frente al CSV congelado. El SHA-256 coincide con `60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1`.

Los recibos locales revisados conservan `PACKAGED_NOT_SUBMITTED`. Esto no acredita ausencia de un envío manual posterior. No disponemos de una sesión autenticada y el inventario de navegadores está vacío. El lector web devolvió contenido ajeno a CodaBench; lo descartamos como evidencia de reglas o clasificación.

El [repositorio oficial](https://github.com/NAIL-HNU/Garl-TTC) exige `submission.json` en la raíz del ZIP, con el esquema `garlttc_prediction_v1`, y remite a CodaBench para evaluar test sin TTC públicos. Antes de enviar, hay que comprobar la participación, cuota e historial de la cuenta y aclarar la admisión del historial causal entre muestras, hasta ocho observaciones y 650 ms. La documentación V16 ya contiene una consulta redactada para organizadores; no hemos enviado mensajes.

La acción externa queda pendiente de autorización expresa separada. No cambiamos pesos, semilla, ensemble, clipping ni valores del paquete para resolver una duda de elegibilidad.

## Comparación congelada propuesta

El [único protocolo nuevo de comparación](FROZEN_COMPARISON_PROTOCOL.json) fija las 630 consultas del manifest FCWD existente, su SHA-256, cinco rutas y seed7. Propone comparar V16, fusión C1 V18, F_TRUE, R_CTX y Garl con sus contratos de entrada declarados. Antes de inferir se deben sellar los adaptadores, endpoints y cobertura de cada ruta y autorizar el presupuesto.

FCWD ya fue observado y carece de la banda negativa de la evaluación firmada completa. Este protocolo estudia transferencia descriptiva; no confirma generalización ciega ni resuelve la regresión negativa. Una comparación confirmatoria requerirá otra población o evaluación oficial elegible. No hemos inferido ninguna ruta del protocolo.

## Refit de una sola familia

El [REFIT_PLAN.json](REFIT_PLAN.json) especifica una cabeza PHASE_ONLY nueva, seed7, sobre H+V, con productores, PAIR, refinadores y normalizadores históricos congelados. H+V contiene 34.854 ejemplos. La propuesta escala 2.500 updates sobre H a 4.730 sobre H+V para conservar la exposición esperada por consulta, y escala el warmup y el coseno. Incluye inicialización, pérdida, masas, roles, genealogía, presupuesto CPU, checkpoints y export.

Este refit parcial amplía TRAIN para la cabeza. No reentrena los productores con TRAIN40 ni sustituye el candidato oficial congelado. Declaramos el stacking secuencial in-sample: los refinadores ya entrenaron en H y sus salidas H no son OOF. H y V dejan de servir como evaluación del nuevo fit. FCWD, Dev32 y test12 quedan fuera del ajuste.

El [adaptador H+V](../../scripts/run_post_v18_refit.py) está implementado. Su [preflight sobre los archivos reales](REFIT_PREFLIGHT.json) verifica genealogía, fuentes, cachés, 34.854 IDs únicos, 16 grupos disjuntos y masas recalculadas. Incluye entrenamiento CPU, checkpoint completo, reanudación con identidad exacta, contabilidad de recuperación y exportación de la cabeza. Las seis pruebas específicas pasan sin llamar al optimizador; comprueban, entre otras cosas, restauración de RNG y sampler, rechazo de corrupción y paridad TorchScript con entradas sintéticas. El bucle de optimización completo permanece sin ejecutar y sin validar mediante entrenamiento.

El comando de entrenamiento rechaza la ejecución sin un recibo separado que vincule la autorización expresa, el hash exacto del plan y su presupuesto. No hemos creado ese recibo. La elección y ejecución del refit requieren una decisión posterior con presupuesto propio. Los límites de RAM y tiempo se comprueban entre operaciones; no son límites impuestos por el sistema operativo. La exportación es una cabeza con contrato fijo de cinco ejemplos, no un servicio de sensores completo.

## Magnitud y saturación de A1-FULL

El [análisis regenerable](A1_NEGATIVE_ANALYSIS.md), su [JSON](A1_NEGATIVE_DIAGNOSTICS.json) y las [2.700 celdas por rango, grupo, signo y saturación](A1_NEGATIVE_RANGE_GROUP.csv) reutilizan las predicciones P. Distinguimos MiDn macro por grupo de MiD pooled y de las métricas condicionales. Conservamos los denominadores y las celdas vacías.

El problema incluye error de magnitud con signo correcto. El análisis empareja consultas donde FULL y A5 aciertan el signo y separa el límite de 60 s del clipping de fase. No deducimos gradientes a partir de pérdidas ni afirmamos haber aislado la causa de la regresión. No ajustamos coeficientes.

## Trabajo V19 que ya existía

La [reconciliación](V19_RECONCILIATION.json) guarda una instantánea de la inscripción y del estado registrado de la rama `scientific-recovery-v19-decision-recovery`. Existe un recibo de continuación posterior a una parada y una rama de entrenamiento `L0/LW`. El estado registrado no certifica que el proceso siga vivo ni que el entrenamiento haya terminado.

Reutilizamos su diagnóstico P tras comprobar el hash del resultado y de las predicciones FULL/A5. No ejecutamos su productor de diagnósticos, que también accede a H y produce otros análisis. No copiamos ni relanzamos su pipeline y no cambiamos sus procesos o archivos.

El único contraste de objetivo que retenemos es el ya registrado L0/LW: mismo padre B1 de correspondencia, FULL, sampler, batch y 12.000 updates por brazo; cambia la masa de la banda negativa. Esta tarea no lo autoriza de nuevo ni consume su presupuesto. No añadimos otro sweep ni un tercer brazo de signo. Las decisiones de esa campaña deben basarse en sus propios resultados y autorizaciones.

## Criterios futuros

Separamos señal frente a un control, elegibilidad para evaluar, promoción del sistema y SOTA oficial. En una evaluación posterior se deben presentar agregado, regresiones por banda, signo, cobertura y coste por separado. El 10 % crucial sigue vigente en el histórico V18. No fijamos ahora otro umbral para reinterpretar las mismas tablas como confirmación.

## Reproducción de esta revisión

En el entorno del repositorio, con `PYTHONPATH=src;.` y `CUDA_VISIBLE_DEVICES=-1` en Windows:

```powershell
python -B scripts/post_v18_review.py prepare
python -B scripts/post_v18_review.py validate
python -B scripts/run_post_v18_refit.py dry-run
```

`prepare` lee artefactos locales de V16, V18 y V19 y escribe únicamente en esta carpeta. Usa CPU para convertir fases guardadas y calcular estadísticas, sin cargar un modelo. Requiere los worktrees hermanos y el entorno con pandas, NumPy y PyTorch; no descarga dependencias. `validate` comprueba la especificación y sus referencias; el `dry-run` del adaptador abre y verifica las cachés reales sin cargar un modelo ni escribir artefactos de entrenamiento. El entorno compartido disponible es `../e-jepa-ttc/.venv/Scripts/python.exe`. [VALIDATION.json](VALIDATION.json) separa estas comprobaciones de las pruebas sintéticas de software y de lo que no se ha ejecutado.
