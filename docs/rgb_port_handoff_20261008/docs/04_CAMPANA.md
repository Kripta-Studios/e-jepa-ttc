# Campaña compacta RGB-PORT: ejecución y decisiones

## P0 — Estado, novedad y verificación

Leer README/NEXT_DECISION nuevos, contabilidad, commit base y propietarios de GPU. Verificar una vez los pesos/cachés usados y sus roles. Consultar registros RGB previos: la existencia de clases/configs no implica un experimento ejecutado; tampoco anunciar como nueva una ejecución exacta ya existente. Limitar esta arqueología a los manifests e informes vinculados, sin recorrer rutas selladas.

Reconciliar los resultados SOTA desde los CSV. No repetir inferencia sobre 3.947 filas ni la reconstrucción TRAIN40 sólo para calcular tablas. El problema Linux LF/Windows CRLF se documenta y se prueba con serialización nueva explícita; no cambiar hashes de los artefactos antiguos.

## P1 — Modelos y datos

Congelar roles P/H/V, manifest de frames/crops/calibración y el esquema RGB. Implementar loader RGB y batch de targets separados. Probar una consulta real TRAIN, forward/backward, gradientes del encoder y guardado/restauración. Máximo 500 updates técnicos entre todas las pruebas, contabilizados aunque se desechen.

Las pruebas de equivalencia event deben demostrar que añadir soporte de datasets RGB no cambia salidas A5/C2F/H8 históricas. No cambiar la clase compartida para otro experimento sin conservar su salida event.

## P2 — Productores

Cuatro IDs prospectivos seed7: E_A5_MATCHED, E_C2F_MATCHED, R_A5, R_C2F. Los dos E pueden reutilizarse sólo con bindings válidos; no convertir la ausencia de reutilización en un permiso para usar TRAIN40 contaminado.

Todos los modelos nuevos aprenden en P durante 18 épocas, endpoint final fijo, batch efectivo32, AdamW, clipping1. LR 3e-4, mínimo3e-5, decay1e-4; primeros tres epochs de warmup de foreground, siguiendo la receta de causal-scale TRAIN40. Preservar las otras constantes de configuración/loss y congelar el JSON expandido antes del primer update.

La autoridad de receta es el JSON efectivo extraído y firmado localmente; una diferencia respecto a esta síntesis se resuelve antes de entrenar, no leyendo V. Primera capa RGB 3 canales, el resto de dimensiones y la familia de transporte se heredan. Pesos nuevos aleatorios seed7; no reshape de un checkpoint event como si fuese una inicialización equivalente.

Usar FP32 para el cálculo de geometría, fase y pérdidas. Conservar precisión ejecutada en la receta base para la CNN; cualquier AMP nuevo necesita admisión TRAIN y se declara, no una afirmación de bit-exactness. Microbatch/accumulation pueden fijarse antes del entrenamiento para encajar; equivalencia de la pérdida y normalización comprobada. El backbone compacto usa GroupNorm, no justificar equivalencia de BatchNorm por acumulación.

Supervisión: misma familia de TTC/log-height ratio/foreground y teacher DINO relacional cuando sus targets RGB están correctamente construidos en P. No aplicar focal/box del event al RGB. Si una etiqueta auxiliar no puede construirse, esa modificación debe predeclararse para controles comparables; no borrarla silenciosamente de un brazo.

No early stopping en V, Dev32 o FCWD. Checkpoint cada100 updates, final fijo y todos los resultados conservados. No entrenar hasta que "gane".

## P3 — PAIR y cabezas

PAIR_E_MATCHED y PAIR_R: encoder A5 correspondiente congelado, 6.840 updates máximo cada una, batch32, LR3e-4, decay1e-4, FP32 y loss de fase conforme al contrato histórico. Sus inputs y suportes pertenecen a su propia modalidad.

Crear features H y V sólo después de congelar ambos productores de cada modalidad y su PAIR. Las cabezas usan datos H, batch128, 2.500 updates, AdamW decay1e-3, warmup100 y cosine 3e-4 →3e-5, clipping1. GRU160, dos capas, output de fase canónico.

Seis brazos de cabeza seed7:

1. E_H1_MATCHED
2. E_CTX_MATCHED
3. R_H1
4. R_CTX
5. F_TRUE
6. F_ZERO

Los refinadores E/R conservan loss histórica de fase +0,1 cuantiles +0,01 coste de expertos de su modalidad. F_TRUE/F_ZERO no tienen selector auxiliar de seis expertos: lambda_cost=0 en ambos, misma loss de fase/cuantiles. La comparación F_TRUE/F_ZERO controla esa receta. Comparar F_TRUE contra E_CTX también es una comparación de sistemas, no atribución pura del componente RGB.

Fusión: dos GRU160 de streams propios, estados concatenados y MLP 320→160→location/width. Inicialización cero de la corrección. Anchor de fases event de productores P, no output de una cabeza entrenada en H. F_ZERO recibe bloque RGB cero, no predicciones RGB residuales ocultas. Mantener el mismo número de pasos/máscaras y tiempos observados para la comparación.

Para desplegar sin RGB, wrapper de fallback a E_CTX inalterado. No prometer que F_TRUE sin RGB y E_CTX sean internamente la misma red; verificar exactamente el wrapper.

Congelar TODOS los endpoints de esta familia antes de su primera evaluación V. No ajustar L, canales, arquitectura, pérdida o coeficientes tras scores de una de las seis cabezas.

## P4 — Evaluación útil, no búsqueda de un número favorable

Primario para eAP V: MiD macro/buckets canónico. Secundarios: MAE, mediana, RTE firmado de la campaña nueva, RMSE, colas p90/p95, error de signo y sobreestimación positiva para TTC<=3s. Si un grupo no tiene targets de un bucket, usar exactamente la regla declarada, no mezclar renormalizaciones.

Contrastes de desarrollo: R_CTX−R_H1; E_CTX−E_H1; F_TRUE−F_ZERO; F_TRUE−E_CTX. Semilla7 es cribado, no confirmación. Bootstrap por grupos completos; intervalos descriptivos y múltiples comparaciones identificadas. No usar ventanas como unidades independientes.

El origen de la mejora se decide por contraste, no por comparar sólo el mejor de varios números. Conservar también los productores RGB puros, PAIR y sus costes para distinguir capacidad de representación de refinamiento posterior.

Una señal práctica para recomendar réplica: mejora MiD >=5% en el contraste primario correspondiente, sin empeorar RTE ni MAE crucial más de5% y sin aumentar error de signo >0,5 puntos porcentuales, con cobertura equivalente. Son umbrales nuevos de cribado, no criterios retrospectivos de aceptación SOTA. Reportar también trade-offs que no pasan este cribado; un valor no significativo no prueba equivalencia.

Evaluar Dev32/FCWD una vez con endpoints congelados como transferencia exploratoria. Mantener original nativo y sensibilidad común ±60 por separado. No convertir el FCWD RGB ausente en MAE=0 ni en un fallo de precisión; sigue `UNAVAILABLE_INPUT_CONTRACT`.

## Presupuesto

Máximo por productor: 18*ceil(N_P/32), N_P<=88.744. Cota máxima 49.932. Cuatro productores: <=199.728. PAIR: <=13.680. Seis cabezas: <=15.000. Total científico <=228.408. Reserva técnica500, recuperación11.092: cap físico240.000. En P~60% de TRAIN40 el trabajo real de productores será menor; calcularlo con N_P efectivo. No reasignar ahorros a nuevos brazos.

## Después, fuera de esta primera autorización

Si el traslado RGB produce evidencia útil: réplica completa de productores+PAIR+cabezas en seeds13/23, no sólo repetir heads. Comparar E_KD versus E_EXTRA al adaptar event con teacher RGB en TRAIN, reconstruyendo PAIR/features/readouts para cada nuevo productor; no reutilizar latentes de otro checkpoint por forma. Refit final/cross-fitting debe aprovechar TRAIN40 sin contaminar evaluación oficial. Registrar ese presupuesto como siguiente decisión, no ejecutarlo del remanente.

Si el compacto RGB pierde: no asumir que la modalidad es inútil. Separar error geométrico, calidad/tiempos y capacidad; entonces estudiar el backbone mayor previamente propuesto con un comparador compacto válido. No lanzar todos los rescates simultáneamente.
