# PHASE-TTC: continuación experimental del 31 de octubre de 2026

Documento regenerado desde recibos de ejecución. Objetivo: mejorar EvTTC sin atribuir a SOTA resultados locales, sin ajustar con las grabaciones del benchmark y comprobando la conservación de eAP.

## Resultado ejecutivo

Se completaron 144 entrenamientos agrupados: 96 en v1b y 48 en v2. Ninguno de los candidatos nuevos pasó todos los criterios registrados. La versión congelada PHASE-TTC-Adapt-20261031 permanece como referencia. Otros seis ajustes del primer lanzamiento quedaron preservados tras un fallo de escritura de progreso; no se contabilizan como una campaña completa.

El trabajo produjo además una corrección verificable de las fuentes del evaluador, la evaluación activa de eAP TRAIN40, una caché Test12 de despliegue sin etiquetas y cuatro aumentos causales del canal geométrico. No se envió correo al benchmark.

## Protocolo y población

Los ajustes usan 2.608 consultas de 25 grabaciones: 2.531 de carretera y 77 de Slider-500. Se excluyen las ocho grabaciones de carretera solapadas y Slider-750/1000. Hay ocho folds que reservan familias completas, con semillas 7,13,21, 300 actualizaciones AdamW, lr=0,01, weight_decay=0,01 y gradiente recortado a 1. Normalización y escala del error absoluto usan exclusivamente el fold de ajuste. La pérdida equilibra familias.

Estos datos tienen exposición histórica. Las métricas OOF evitan ajuste directo sobre el grupo evaluado en este experimento, pero no constituyen una reserva nueva. La selección exige cobertura completa y no empeorar MAE ni error relativo macro por grabación, tanto en carretera como en Slider-500, frente a IQ. No se abrió el benchmark para escoger los candidatos; al rechazarse todos, no se generaron nuevas entregas.

## Controles entrenados

IQ reproduce la fusión inversa QA anterior bajo los mismos folds. LG selecciona signo y mezcla magnitudes logarítmicas; LA añade perturbaciones de expertos. GR corrige la geometría por un factor entre 0,5 y 2 sobre la entrada limitada a ±60 s. La revisión detectó que el promedio entre semillas podía volver a cancelar signos, y que una decisión de signo discreta carecía de gradiente directo en algunos casos.

SG añade supervisión de fiabilidad exclusivamente durante entrenamiento: coste esperado de cada experto, limitado a 10, ponderado por 0,5. SGA añade aumentos del canal geométrico. Ambos usan mediana de tres semillas y reaplican el retorno exacto a F0 cuando no existe historia válida. Se conservan predicciones OOF de cada semilla. Es una comparación de sistemas; cambia también la agregación respecto a IQ.

| Campaña/control | RAE carretera macro (%) | MAE carretera macro (s) | RAE Slider-500 (%) | MAE Slider-500 (s) |
|---|---:|---:|---:|---:|
| v1b/F0 | 30.6004 | 0.810620 | 83.8874 | 1.178208 |
| v1b/GR | 24.8917 | 0.771455 | 19.4199 | 0.380000 |
| v1b/IQ | 17.7489 | 0.543418 | 20.8519 | 0.317744 |
| v1b/LA | 18.1358 | 0.567442 | 28.2708 | 0.489568 |
| v1b/LG | 18.6083 | 0.596158 | 34.1066 | 0.594602 |
| v1b/geometry_F0 | 35.1905 | 1.129698 | 19.9217 | 0.425228 |
| v2/SG | 17.6366 | 0.562741 | 22.4804 | 0.423939 |
| v2/SGA | 21.7806 | 0.672036 | 20.9875 | 0.384191 |

Selección v1b: `None`. Selección v2: `None`.

## Aumentos realmente ejecutados

Se recomputaron 10.432 combinaciones consulta/variante: últimas dos cajas, últimas cuatro, jitter independiente de ±2 píxeles y pérdida del 50 % de cajas. La geometría limpia y las 20 variables de calidad reproducen exactamente la caché original en las 2.608 consultas antes de generar variantes. Cada perturbación conserva causalidad, historia máxima de 650 ms y la identidad original. Native y targets permanecen iguales.

Este experimento perturba solamente las observaciones del estimador geométrico. No ejecuta de nuevo los productores sobre sensores perturbados ni reproduce un arranque en frío de toda la cadena. SGA no pasó los criterios: no se presenta dataset augmentation como una mejora demostrada.

## Corrección del evaluador

Los enlaces de ground truth de CCRs1-low y CCRs1-medium en la página oficial corresponden a los archivos que localmente estaban asignados al revés. Se guardaron URL, bytes y hashes; se mantuvieron los offsets demostrados por identidad RGB. La reasignación reproduce 7,5688559 % y 9,4237158 % para la entrega original y recupera las once etiquetas ausentes. Ahora hay 434 targets públicos alineados.

[Fuente oficial de descargas](https://nail-hnu.github.io/EvTTC/download/download/). Los ocho resultados de carretera y Slider-1000 reproducen el PDF a su precisión publicada. Esto no prueba igualdad del vector privado de targets. Slider-750 sigue discrepando: el CSV público da 283,931058 % para el original procesado y el PDF 132 %. No se ajustó ningún offset ni transformación para forzar ese resultado.

| Población | Original procesado RAE (%) | Versión congelada actual RAE (%) |
|---|---:|---:|
| all434_public_diagnostic | 49.894242 | 17.249803 |
| road365 | 11.917132 | 7.469419 |
| pdf_reproduced_sequences383 | 18.730071 | 9.417157 |
| Slider-750 | 283.931058 | 76.071442 |
| Slider-1000 | 156.881341 | 48.912957 |

La tabla reevalúa modelos ya existentes. No atribuye estas diferencias a los nuevos ajustes rechazados. El agregado público de 434 consultas no es el score oficial mientras Slider-750 no esté resuelto.

## Conservación de eAP y Test12

Se construyó historia causal por consulta y por objeto a partir de los metadatos P/H/V firmados: 88.744 ejemplos, 40 secuencias y 1.812 tracks. Todas las consultas activan la fusión; 88.597 tienen geometría válida. Así se comprueba la rama activa, en vez de repetir únicamente la identidad del fallback sin cajas.

| eAP TRAIN40 | MAE (s) | Error relativo (%) |
|---|---:|---:|
| F0 | 1.416030 | 27.365738 |
| current_active | 1.177047 | 22.195639 |

Estas son métricas de conservación/resustitución TRAIN40 con features upstream OOF; la cabeza original ya se ajustó con TRAIN40. No son generalización independiente. La futura promoción exige no empeorar la referencia adaptada activa.

Test12 aporta 6.762 consultas lógicas, 5.805 físicas y 12 secuencias, con estados reales de despliegue de 340 dimensiones y las cajas causales. Se verificaron archivos, aliases, sellos y reproducción de F0; el error máximo de TTC al repetir la cabeza es 4,577637e-5 s, dentro de la tolerancia registrada, no igualdad bit a bit. No hay targets Test12 disponibles en las fuentes locales auditadas. Por eso esta caché permite medir deriva de predicciones, no afirmar precisión ni ausencia de regresión.

## Próxima hipótesis y límites

Se abre un experto físico de expansión de eventos con referencia al final de la ventana. Una caja actual puede delimitar eventos pasados; no permite por sí sola medir expansión de cajas. Primero se exige recuperación sintética y luego desarrollo en Slider-500. No equivale a reproducir el CMax del leaderboard: no se encontró su solver exacto. El código público STRTTC usa convenciones temporales/ROI diferentes.

Ningún resultado de esta continuación demuestra SOTA. Sigue pendiente la confirmación del evaluador Slider-750 y una comparación equivalente. Se preservan los resultados negativos y la versión previa, sin sustituirla por una variante que solo mejore una métrica.

## Reproducción

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/report_phase_safe_research.py
```

Los scripts train_phase_safe_fusion.py y train_phase_safe_supervised.py documentan sus argumentos con --help. Cada campaña conserva CONFIG, código ejecutado, hashes, checkpoints, pérdidas y grupos de ajuste. Use directorios nuevos para repetir; las campañas completadas no se sobreescriben.
