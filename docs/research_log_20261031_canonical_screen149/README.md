# PHASE-TTC: screen fijo de PAIR canónico, 149 consultas

Desarrollo con exposición histórica: 72 consultas de carretera y 77 de Slider-500. No es evaluación oficial, confirmación independiente ni evidencia de SOTA.

La extracción y los replays tardaron **999.372 s**. Las 149 consultas reprodujeron exactamente los arrays originales disponibles de productores, contextos y cabeza antes de aplicar la corrección. Se conservaron los lotes, los relojes, los normalizadores y los pesos; no hubo entrenamiento.

Se capturó PAIR133 y se reconstruyó raw17 completo, incluidos sus desacuerdos, antes de recalcular contextos y cabeza. El router es el adaptado de FINAL_MODEL, con sus mismas entradas de geometría/calidad ligadas al hash de la población. F0 y router se evalúan por separado. Predicciones y ausencias se sellaron antes de que otro proceso abriera targets.

## Precisión y cobertura

MAE y error relativo son medias macro por grabación en el conjunto emparejado. Se muestran todos los denominadores; una pérdida de cobertura invalida el gate aunque mejoren esas métricas.

| Ruta | Dominio | Disponibles original → canónico | Población | MAE original → canónico (s) | Error relativo original → canónico (%) | No empeora |
|---|---|---:|---:|---:|---:|---|
| F0 | road | 72 → 72 | 72 | 0.748853 → 0.748672 | 31.370683 → 31.358111 | True |
| F0 | slider | 76 → 76 | 77 | 1.178156 → 1.120809 | 83.885894 → 82.442047 | True |
| router | road | 72 → 72 | 72 | 0.338041 → 0.338051 | 10.295071 → 10.295517 | False |
| router | slider | 77 → 77 | 77 | 0.205066 → 0.198997 | 13.578564 → 13.426316 | True |

## Rechazos y comprobaciones

- canonical: 578 observaciones de eventos y 838 RGB aceptadas; 1 lotes rechazados.
- original: 578 observaciones de eventos y 838 RGB aceptadas; 1 lotes rechazados.
- F0/road: 0 nuevas ausencias; 0 cambios de signo.
- F0/slider: 0 nuevas ausencias; 0 cambios de signo.
- router/road: 0 nuevas ausencias; 0 cambios de signo.
- router/slider: 0 nuevas ausencias; 0 cambios de signo.

En las dos consultas comunes, el replay compacto reproduce exactamente los raw17/expert3 y los contextos/cabeza canónicos del piloto anterior, que sí ejecutaba los productores completos. La auditoría compara artefactos sellados; no repite los pilotos ni abre targets.

## Criterio fijado y resultado

Antes de abrir targets se exigió conservar cobertura, no empeorar ninguno de los cuatro criterios (MAE/error relativo × carretera/Slider) y mejorar al menos uno, por cada ruta. No se cambia d0 ni se elige otra corrección a partir de estos resultados.

- F0: justifica evaluar la población completa.
- router: no supera el screen; no promover ni ampliar esta ruta como candidato aprobado.

## Resultados por grabación

| Ruta | Grabación | Pares/población | MAE original → canónico (s) | Error relativo original → canónico (%) |
|---|---|---:|---:|---:|
| F0 | CCRm-low-0-overlap-0 | 3/3 | 0.399650 → 0.399652 | 12.403879 → 12.403905 |
| F0 | CCRm-low-50-overlap-50 | 3/3 | 0.435191 → 0.435212 | 13.012053 → 13.012564 |
| F0 | CCRm-medium-0-overlap-0 | 3/3 | 0.645471 → 0.645453 | 9.826449 → 9.826174 |
| F0 | CCRm-medium-50-overlap-50 | 3/3 | 0.493783 → 0.493788 | 14.265790 → 14.265956 |
| F0 | CCRs-1-high-0-overlap-0 | 3/3 | 1.671386 → 1.670319 | 51.483719 → 51.448694 |
| F0 | CCRs-1-high-50-overlap-50 | 3/3 | 0.268587 → 0.268924 | 13.051056 → 13.069032 |
| F0 | CCRs-1-low-0-overlap-0 | 3/3 | 0.369031 → 0.369050 | 7.260291 → 7.260594 |
| F0 | CCRs-1-low-50-overlap-50 | 3/3 | 0.037527 → 0.037500 | 1.569471 → 1.568456 |
| F0 | CCRs-1-medium-0-overlap-0 | 3/3 | 0.200212 → 0.200210 | 8.353772 → 8.353613 |
| F0 | CCRs-1-medium-50-overlap-50 | 3/3 | 0.066007 → 0.066019 | 4.142960 → 4.144014 |
| F0 | CCRs-3-low-100-overlap-100 | 3/3 | 0.256205 → 0.256203 | 8.630097 → 8.629946 |
| F0 | CCRs-3-medium-100-overlap-100 | 3/3 | 0.102723 → 0.102730 | 5.670214 → 5.670452 |
| F0 | CCRs-side-high | 3/3 | 0.558745 → 0.559207 | 29.743619 → 29.778169 |
| F0 | CCRs-side-low | 3/3 | 1.549392 → 1.549465 | 50.520793 → 50.522198 |
| F0 | CCRs-side-medium | 3/3 | 1.668041 → 1.668097 | 71.818494 → 71.819885 |
| F0 | CPLA-high | 3/3 | 0.561666 → 0.557133 | 33.202758 → 32.863891 |
| F0 | CPLA-low | 3/3 | 0.694200 → 0.694265 | 19.508997 → 19.510370 |
| F0 | CPLA-medium | 3/3 | 1.006248 → 1.006291 | 43.942641 → 43.945759 |
| F0 | CPNA-high | 3/3 | 0.874886 → 0.874929 | 61.442594 → 61.446183 |
| F0 | CPNA-low | 3/3 | 1.011727 → 1.011806 | 26.009627 → 26.011308 |
| F0 | CPNA-medium | 3/3 | 1.162299 → 1.162330 | 85.216296 → 85.219329 |
| F0 | CPNAO-high | 3/3 | 2.339702 → 2.339736 | 95.437229 → 95.438478 |
| F0 | CPNAO-low | 3/3 | 1.103898 → 1.103909 | 44.852493 → 44.852711 |
| F0 | CPNAO-medium | 3/3 | 0.495888 → 0.495907 | 41.531108 → 41.532973 |
| F0 | Slider-500 | 76/77 | 1.178156 → 1.120809 | 83.885894 → 82.442047 |
| router | CCRm-low-0-overlap-0 | 3/3 | 0.206233 → 0.206233 | 6.341066 → 6.341066 |
| router | CCRm-low-50-overlap-50 | 3/3 | 0.462373 → 0.462375 | 12.408328 → 12.408385 |
| router | CCRm-medium-0-overlap-0 | 3/3 | 1.115568 → 1.115565 | 16.898715 → 16.898646 |
| router | CCRm-medium-50-overlap-50 | 3/3 | 0.412774 → 0.412769 | 9.294865 → 9.294784 |
| router | CCRs-1-high-0-overlap-0 | 3/3 | 0.646162 → 0.646032 | 20.618775 → 20.614433 |
| router | CCRs-1-high-50-overlap-50 | 3/3 | 0.314086 → 0.314089 | 9.906578 → 9.907092 |
| router | CCRs-1-low-0-overlap-0 | 3/3 | 0.516790 → 0.516796 | 11.527583 → 11.527697 |
| router | CCRs-1-low-50-overlap-50 | 3/3 | 0.209795 → 0.209792 | 8.078647 → 8.078513 |
| router | CCRs-1-medium-0-overlap-0 | 3/3 | 0.077629 → 0.077630 | 3.064148 → 3.064240 |
| router | CCRs-1-medium-50-overlap-50 | 3/3 | 0.125268 → 0.125268 | 5.035023 → 5.035123 |
| router | CCRs-3-low-100-overlap-100 | 3/3 | 0.165790 → 0.165790 | 5.918818 → 5.918818 |
| router | CCRs-3-medium-100-overlap-100 | 3/3 | 0.083495 → 0.083495 | 4.800959 → 4.800974 |
| router | CCRs-side-high | 3/3 | 0.099338 → 0.099316 | 5.225150 → 5.223254 |
| router | CCRs-side-low | 3/3 | 0.049385 → 0.049384 | 1.344031 → 1.344028 |
| router | CCRs-side-medium | 3/3 | 0.704959 → 0.704965 | 14.078999 → 14.079095 |
| router | CPLA-high | 3/3 | 0.124474 → 0.124692 | 8.949431 → 8.960316 |
| router | CPLA-low | 3/3 | 0.616872 → 0.616906 | 15.606738 → 15.607544 |
| router | CPLA-medium | 3/3 | 0.348264 → 0.348292 | 16.440703 → 16.442159 |
| router | CPNA-high | 3/3 | 0.099439 → 0.099432 | 8.323143 → 8.322551 |
| router | CPNA-low | 3/3 | 1.082558 → 1.082639 | 28.132566 → 28.134276 |
| router | CPNA-medium | 3/3 | 0.221591 → 0.221589 | 13.405122 → 13.404957 |
| router | CPNAO-high | 3/3 | 0.044667 → 0.044669 | 2.079782 → 2.079804 |
| router | CPNAO-low | 3/3 | 0.310487 → 0.310517 | 12.587469 → 12.588736 |
| router | CPNAO-medium | 3/3 | 0.074983 → 0.074993 | 7.015069 → 7.015917 |
| router | Slider-500 | 77/77 | 0.205066 → 0.198997 | 13.578564 → 13.426316 |

## Límites y reproducción

La consulta Slider-500:76 conserva su ausencia nativa por fin de cobertura de eventos. La disponibilidad del router puede diferir porque dispone de geometría. No se ha modificado ni sustituido el entregable.

El piloto anterior conserva la regresión del contexto de eventos de carretera al ralentizar el tiempo. La auditoría eAP de observaciones actuales no equivale a conservación de historias, contextos y cabeza completos. La discrepancia oficial de Slider-750 sigue abierta; no se ha enviado correo ni otra entrega.

Inferencia local sellada: `output\phase_timewarp\screen149_v1`. Evaluación: `output\phase_timewarp\screen149_evaluation_v1`. Los NPZ y pesos siguen fuera de Git. Los JSON pequeños publicados normalizan CRLF a LF; los hashes de recibos describen los bytes locales originales.

```powershell
.venv-transfer/Scripts/python.exe scripts/report_phase_canonical_screen.py --screen output\phase_timewarp\screen149_v1 --evaluation output\phase_timewarp\screen149_evaluation_v1 --compact-audit output\phase_timewarp\screen149_audit_v1\COMPACT_PARITY.json
```
