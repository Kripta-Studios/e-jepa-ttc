# PHASE-TTC: experto causal de expansión de eventos

Informe regenerado desde ejecuciones del 31 de octubre de 2026. Este experto es una hipótesis física adicional; todavía no sustituye al modelo congelado.

## Qué se implementó

Un estimador original de tres parámetros q,bx,by sobre eventos causales dentro de una ROI actual. La transformación p_ref = p + (q p − b)(t_ref − t) refiere q=1/TTC al instante de consulta. El foco de expansión es libre; no se impone que coincida con el centro de la caja. Se conservan polaridades separadas. El solver usa una rejilla de 81 candidatos, tres inicios Powell y 240 evaluaciones por inicio, con límites y criterios de ausencia explícitos.

Las referencias matemáticas son [contraste de eventos](https://arxiv.org/abs/1804.01306) y [STRTTC](https://arxiv.org/abs/2407.07324). No se copió código externo. No se afirma reproducir el CMax de EvTTC: no se encontró su solver exacto; la entrada pública STRTTC usa otra referencia temporal y preparación de ROI.

## Pruebas y resultados negativos

La energía del histograma deformado favoreció contracciones espurias en escenas estacionarias. Perfiles con traslación conocida y cambios de raster demostraron sesgo del objetivo; aumentar solamente el presupuesto no lo corregía. También se probó una referencia interna al punto medio, con conversión exacta al endpoint, y se descartó una compensación por Jacobiano tras su diagnóstico.

| Receta | Casos que pasan | Casos ejecutados |
|---|---:|---:|
| energy_original | 8 | 18 |
| energy_midpoint | 3 | 6 |
| ncc_development | 6 | 6 |
| ncc_confirmation | 12 | 12 |

La receta aceptada compara imágenes de las mitades temprana y tardía mediante correlación normalizada de media cero. Mantiene el warp al endpoint, el raster128², Gaussianσ0,8, los límites y el presupuesto. Añade correlación mínima 0,8 como requisito conservador; no relaja las tolerancias de recuperación. La confirmación usa semillas31/47, que no se habían abierto al elegir esta receta.

Son contornos sintéticos ideales, no un simulador fotométrico completo de cámara de eventos. Superar estas pruebas permite investigar datos reales, no demostrar SOTA.

## Mezcla de movimientos: sensibilidad al fondo

Una prueba posterior fija las semillas nuevas 61 y 73 y añade un contorno estático independiente, manteniendo exactamente los eventos originales del objeto y su TTC. Se prueban proporciones de fondo 0,25,50,75 % del total, con el mismo solver, límites y criterios de aceptación. La máscara sintética de origen se usa solamente para verificar el fixture; nunca entra al estimador.

| Semilla | Fondo (%) | q verdadero (1/s) | q estimado (1/s) | Válido | Recupera |
|---|---:|---:|---:|---|---|
| 61 | 0 | 2.341091 | 2.372093 | True | True |
| 61 | 25 | 2.341091 | 2.287154 | True | True |
| 61 | 50 | 2.341091 | 0.770587 | True | False |
| 61 | 75 | 2.341091 | 0.089606 | False | False |
| 73 | 0 | 2.843704 | 2.834703 | True | True |
| 73 | 25 | 2.843704 | 2.745855 | True | True |
| 73 | 50 | 2.843704 | 1.141274 | True | False |
| 73 | 75 | 2.843704 | 0.112594 | False | False |

Con 50 % de fondo, ambos casos producen una estimación finita aceptada pero incorrecta. Con 75 % el solver se abstiene. Esto demuestra una limitación del criterio de confianza frente a movimientos mezclados; no identifica la causa del error real de Slider ni justifica recortar su ROI usando targets. No se ajustó el solver tras observar estos ocho resultados.

## Slider-500 real

La caché conserva77consultas y889.637eventos, ventana causal de100ms recortada a la cobertura disponible, ROIactual ampliada1,2× y hasta20.000eventos por consulta. El primer instante tiene226eventos; el último conserva su consulta aunque la cobertura termine262µs antes. Se registran cámaras, reloj, selección y calibración. Se corrige distorsión radtan antes del solver. En los bordes se ampliaron las iteraciones de inversión de12hasta64 para cumplir la tolerancia ya fijada de 0,001píxeles; no se modificaron coeficientes ni targets. El intento parcial previo de74consultas se conserva.

Las77predicciones se sellaron antes de abrir los targets. Slider-500 pertenece a desarrollo y la cabeza adaptada de referencia ya se ajustó con esta grabación. No es una comparación independiente ni una puntuación del benchmark.

| Estimador/población | Cobertura | RAE (%) | MAE (s) |
|---|---:|---:|---:|
| Eventos, condicionado a disponibilidad | 62/77 | 15.693702 | 0.268963 |
| Geometría en las mismas62consultas | 62/62 | 14.711172 | 0.207334 |
| Adaptador actual,77consultas | 77/77 | 13.577908 | 0.205044 |
| Eventos disponibles; si no, adaptador actual | 77/77 | 13.876810 | 0.256119 |
| Eventos; si no, geometría; si no, F0 | 77/77 | 20.712790 | 0.474851 |

El experto de eventos gana a la geometría en 36/62 consultas compartidas, pero sus errores extremos hacen que la sustitución directa no mejore el sistema completo. En las primeras10consultas se abstiene10veces: esta implementación todavía no resuelve el arranque. Un score condicionado no sustituye a cobertura completa.

CPU: media 1.658280s/consulta; p95 2.778245s. Incluye lectura de caché, desdistorsión, solver y recibo; excluye extracción de sensores crudos.

## GPU, precarga y CUDA Graphs

Se implementó un objetivo PyTorch separado con precarga de puntos, edades, polaridad y bins temporales; acumulación bilineal; y padding gaussiano que reproduce el reflect de SciPy. Se capturaron grafos reales para lotes de1y81candidatos. La caché del solver mantiene una sola forma de eventos para limitar VRAM.

Float64 reproduce el objetivo de referencia. Float32 se rechaza: el redondeo cerca del borde cambia admisión/pérdida de eventos y viola la tolerancia de1e-6. No se presenta mixed precision como una optimización válida para este solver.

La comparación integrada abarca18casos sintéticos y5consultas reales sin etiquetas. Validez y causa coinciden exactamente, y q/b están dentro de1e-5. Cinco casos sintéticos cambian el número de evaluaciones Powell: equivalencia numérica no significa reproducción bit a bit ni del camino del optimizador.

Los recibos objective_profile.json y accelerator.json contienen tiempos, memoria, capturas y replays medidos. El benchmark mide proceso CUDA inicializado y separa captura de grafos y reutilización; no confundirlo con arranque frío de la aplicación.

La ampliación posterior a las 77 consultas falló la puerta estricta: 76/77 cumplen la tolerancia q/b de 1e-5. Las decisiones de validez y rechazo coinciden en las 77. Slider-500:15 es rechazado por ambos solvers como flat_objective, pero sus parámetros internos difieren más de lo permitido. Se conserva el resultado fallido; no se amplía retrospectivamente la tolerancia.

Esta pasada tardó 25.893 s y reservó como máximo 429.916 MB. Hubo comprobaciones CPU breves simultáneas al principio: estos tiempos son observaciones de ejecución, no un benchmark aislado ni una prueba de aceleración del sistema PHASE completo. La comparación no abrió targets.

Se registró después una política híbrida nueva: recalcular en CPU toda abstención de GPU y conservar su resultado exacto, sin consultar etiquetas. La comprobación exige además TTC finito coherente con 1/q para resultados válidos, ausencia de TTC en rechazos y tolerancia en segundos derivada de la diferencia de q. No impone una tolerancia absoluta en segundos ajena a la coordenada original. El fallo previo permanece conservado.

La pasada híbrida cumple 77/77 comparaciones con 15 retornos a CPU. RUN_CONTEXT registra la coordinación para evitar otros benchmarks de esta tarea, aunque no descarta actividad de fondo del sistema. No se ejecutó una referencia CPU contemporánea para medir una aceleración controlada.

En esta segunda pasada también pasaron todas las estimaciones GPU antes del retorno a CPU: el caso fallido anterior muestra variación numérica del camino de Powell, no un fallo determinista que haya desaparecido. Estas pruebas no garantizan paridad universal ni ejecución bit a bit. La variante sigue siendo una herramienta experimental separada.

## Decisión

No se promueve una nueva entrega. Se conserva el experto como candidato a aportar información complementaria y se continúa investigando arranque y errores correlacionados. La precisión de eAP no se altera por añadir este módulo experimental separado. Cualquier integración requiere un nuevo protocolo, ajuste por grupos, control de conservación de eAP y evaluación con cobertura completa.

## Reproducir este informe

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/report_phase_event_expansion.py
```
