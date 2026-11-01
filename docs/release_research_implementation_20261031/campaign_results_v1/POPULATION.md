# PHASE-TTC: PAIR canónico en la población de 2.608 consultas

F0 supera los criterios fijados; procede comprobar eAP completo antes de promoverlo.

Desarrollo con exposición histórica: 2.531 consultas de carretera en 24 grabaciones y 77 de Slider-500. No es evaluación oficial ni confirmación independiente de SOTA. El router ya falló el screen previo y sus resultados aquí son solo diagnósticos.

Tiempo de ejecución: 13243.949 s. Se reutilizaron las 149 consultas selladas del screen y se extrajeron 2.459 nuevas. Cada consulta nueva reprodujo exactamente su ejecución original en PAIR, raw17, contextos y cabeza. El control es la ejecución original emparejada; las características nativas de la antigua POPULATION.npz no son el control de este experimento.

Se conservaron pesos, normalizadores, relojes y lotes. Se reconstruyó raw17 completo a partir de PAIR133 y se recalcularon contextos/cabeza. No hubo actualizaciones del optimizador. Las predicciones y ausencias se sellaron antes de unir targets en otro proceso.

## Precisión y cobertura

Métricas macro por grabación en el conjunto emparejado, con salidas firmadas sin recorte y denominador relativo max(abs(target), 1e-6). Los criterios exigen conservar cobertura, no empeorar MAE ni error relativo en ninguno de los dos dominios y mejorar al menos uno. No se reajustan umbrales tras ver el resultado.

| Ruta | Dominio | Original → canónico | Pares/targets/población | MAE original → canónico (s) | Error relativo original → canónico (%) |
|---|---|---:|---:|---:|---:|
| F0 | road | 2531 → 2531 | 2531/2531/2531 | 0.810620018 → 0.810489612 | 30.600389999 → 30.591045605 |
| F0 | slider | 76 → 76 | 76/77/77 | 1.178155898 → 1.120808950 | 83.885893960 → 82.442047433 |
| router | road | 2531 → 2531 | 2531/2531/2531 | 0.395520983 → 0.395511332 | 11.157105075 → 11.156679818 |
| router | slider | 77 → 77 | 77/77/77 | 0.205066178 → 0.198997120 | 13.578564039 → 13.426316126 |

## Cambios por grabación y bootstrap

Bootstrap pareado de grabaciones de carretera (10.000 remuestreos, semilla 7). Diferencia canónico − original: negativa indica menor error. Es descriptivo; no trata ventanas correlacionadas como independientes. Slider tiene una sola grabación y no recibe un intervalo entre grabaciones.

| Ruta | Métrica | Cambio medio | IC 95% |
|---|---|---:|---|
| F0 | mae_seconds | -0.000130406 | [-0.000504700, 0.000093150] |
| F0 | relative_error_pct | -0.009344394 | [-0.035002205, 0.005325885] |
| router | mae_seconds | -0.000009650 | [-0.000032949, 0.000009215] |
| router | relative_error_pct | -0.000425257 | [-0.001369364, 0.000202919] |

| Ruta | Grabación | Pares/población | MAE original → canónico (s) | Error relativo original → canónico (%) |
|---|---|---:|---:|---:|
| F0 | CCRm-low-0-overlap-0 | 107/107 | 0.518605915 → 0.518598148 | 17.035682698 → 17.035517043 |
| F0 | CCRm-low-50-overlap-50 | 140/140 | 0.923657552 → 0.923657502 | 17.564298982 → 17.564279924 |
| F0 | CCRm-medium-0-overlap-0 | 159/159 | 1.320366263 → 1.320355701 | 13.017783318 → 13.017631707 |
| F0 | CCRm-medium-50-overlap-50 | 174/174 | 1.341267235 → 1.341257865 | 19.327812499 → 19.327522325 |
| F0 | CCRs-1-high-0-overlap-0 | 87/87 | 0.678029502 → 0.677567561 | 20.947011906 → 20.933541787 |
| F0 | CCRs-1-high-50-overlap-50 | 132/132 | 0.471866714 → 0.472212916 | 16.583209340 → 16.600340134 |
| F0 | CCRs-1-low-0-overlap-0 | 112/112 | 0.422843926 → 0.422850750 | 9.293440602 → 9.293601990 |
| F0 | CCRs-1-low-50-overlap-50 | 96/96 | 0.108226368 → 0.108233935 | 5.068350145 → 5.068799028 |
| F0 | CCRs-1-medium-0-overlap-0 | 83/83 | 0.176901278 → 0.176895129 | 7.267097918 → 7.266903297 |
| F0 | CCRs-1-medium-50-overlap-50 | 83/83 | 0.122541173 → 0.122541484 | 6.655002781 → 6.655238508 |
| F0 | CCRs-3-low-100-overlap-100 | 149/149 | 0.358076817 → 0.358074592 | 8.495575625 → 8.495500818 |
| F0 | CCRs-3-medium-100-overlap-100 | 101/101 | 0.396435497 → 0.396435306 | 13.104191758 → 13.104070942 |
| F0 | CCRs-side-high | 104/104 | 0.515585055 → 0.516281522 | 24.026194449 → 24.064593511 |
| F0 | CCRs-side-low | 138/138 | 1.319683892 → 1.319706060 | 44.735240725 → 44.735941881 |
| F0 | CCRs-side-medium | 137/137 | 1.525643852 → 1.525664396 | 50.278894298 → 50.280021161 |
| F0 | CPLA-high | 70/70 | 0.888660849 → 0.884702840 | 51.740258467 → 51.458105450 |
| F0 | CPLA-low | 145/145 | 0.734846127 → 0.734853146 | 26.270909850 → 26.271313408 |
| F0 | CPLA-medium | 78/78 | 0.832970041 → 0.832983718 | 47.072163160 → 47.073266113 |
| F0 | CPNA-high | 60/60 | 0.994250493 → 0.994335968 | 63.763032502 → 63.768818605 |
| F0 | CPNA-low | 117/117 | 1.044916360 → 1.044940979 | 33.277713354 → 33.278636176 |
| F0 | CPNA-medium | 55/55 | 1.068478776 → 1.068498602 | 70.022138920 → 70.023483646 |
| F0 | CPNAO-high | 100/100 | 2.134012171 → 2.134043230 | 80.642097675 → 80.643404574 |
| F0 | CPNAO-low | 70/70 | 1.083325596 → 1.083346152 | 47.835944342 → 47.837051281 |
| F0 | CPNAO-medium | 34/34 | 0.473688986 → 0.473713186 | 40.385314664 → 40.387511219 |
| F0 | Slider-500 | 76/77 | 1.178155898 → 1.120808950 | 83.885893960 → 82.442047433 |
| router | CCRm-low-0-overlap-0 | 107/107 | 0.343925709 → 0.343924639 | 12.381812370 → 12.381827770 |
| router | CCRm-low-50-overlap-50 | 140/140 | 0.701431624 → 0.701432821 | 12.078820862 → 12.078837815 |
| router | CCRm-medium-0-overlap-0 | 159/159 | 1.221877372 → 1.221872139 | 12.859510526 → 12.859499602 |
| router | CCRm-medium-50-overlap-50 | 174/174 | 1.167901547 → 1.167894461 | 15.904812572 → 15.904647466 |
| router | CCRs-1-high-0-overlap-0 | 87/87 | 0.291521678 → 0.291418239 | 10.476074790 → 10.474137723 |
| router | CCRs-1-high-50-overlap-50 | 132/132 | 0.341709004 → 0.341814061 | 9.986646062 → 9.988608190 |
| router | CCRs-1-low-0-overlap-0 | 112/112 | 0.271998527 → 0.271999245 | 6.181204302 → 6.181236295 |
| router | CCRs-1-low-50-overlap-50 | 96/96 | 0.196434105 → 0.196432490 | 7.625318814 → 7.625427145 |
| router | CCRs-1-medium-0-overlap-0 | 83/83 | 0.085685894 → 0.085684926 | 3.713171319 → 3.713131591 |
| router | CCRs-1-medium-50-overlap-50 | 83/83 | 0.128398219 → 0.128398361 | 5.509205085 → 5.509186430 |
| router | CCRs-3-low-100-overlap-100 | 149/149 | 0.287565265 → 0.287565735 | 7.100317834 → 7.100291127 |
| router | CCRs-3-medium-100-overlap-100 | 101/101 | 0.364702090 → 0.364706453 | 11.553791703 → 11.553883995 |
| router | CCRs-side-high | 104/104 | 0.145727995 → 0.145684622 | 6.119740754 → 6.117331927 |
| router | CCRs-side-low | 138/138 | 0.408185879 → 0.408186387 | 8.880861788 → 8.880888105 |
| router | CCRs-side-medium | 137/137 | 0.439071941 → 0.439071154 | 6.930338401 → 6.930324735 |
| router | CPLA-high | 70/70 | 0.155091439 → 0.154879055 | 9.891073799 → 9.881422439 |
| router | CPLA-low | 145/145 | 0.386149360 → 0.386148524 | 11.252023229 → 11.252009219 |
| router | CPLA-medium | 78/78 | 0.274380235 → 0.274382189 | 12.260820379 → 12.260916460 |
| router | CPNA-high | 60/60 | 0.159452356 → 0.159456329 | 10.417663213 → 10.417840295 |
| router | CPNA-low | 117/117 | 0.599892415 → 0.599900545 | 16.378403148 → 16.378688454 |
| router | CPNA-medium | 55/55 | 0.343230159 → 0.343234041 | 17.538787760 → 17.538978148 |
| router | CPNAO-high | 100/100 | 0.660271105 → 0.660276182 | 24.228501448 → 24.228689477 |
| router | CPNAO-low | 70/70 | 0.379026881 → 0.379032465 | 16.509018748 → 16.509483082 |
| router | CPNAO-medium | 34/34 | 0.138872785 → 0.138876910 | 11.992602893 → 11.993028146 |
| router | Slider-500 | 77/77 | 0.205066178 → 0.198997120 | 13.578564039 → 13.426316126 |

## Disponibilidad y decisiones

- canonical: 10414 observaciones de eventos, 13133 RGB; 1 lotes rechazados.
- original: 10414 observaciones de eventos, 13133 RGB; 1 lotes rechazados.
- F0: cumple criterios numéricos fijos = True.
- F0/road: 0 nuevas ausencias, 0 nuevas disponibilidades y 0 cambios de signo.
- F0/slider: 0 nuevas ausencias, 0 nuevas disponibilidades y 0 cambios de signo.
- router: cumple criterios numéricos fijos = True.
- router/road: 0 nuevas ausencias, 0 nuevas disponibilidades y 0 cambios de signo.
- router/slider: 0 nuevas ausencias, 0 nuevas disponibilidades y 0 cambios de signo.

## Límites y reproducción

El entregable sigue intacto. El resultado del router no revoca su fallo previo. La auditoría parcial eAP de consultas actuales no demuestra conservación de historias, contextos y cabeza; sus cachés OOF tampoco prueban preservación de despliegue. Se conserva la regresión temporal del contexto de eventos del piloto. Sigue pendiente la discrepancia oficial de Slider-750.

Inferencia: `output\phase_timewarp\population2608_v1`. Evaluación: `output\release_research_20261031\continuation_v1\population_evaluation`. Se publican el informe JSON, el protocolo y un recibo compacto con hashes; los arrays, pesos y el listado completo de hashes por consulta permanecen localmente. Los JSON copiados normalizan CRLF a LF; los recibos conservan hashes de los bytes locales.

```powershell
.venv-transfer/Scripts/python.exe scripts/report_phase_canonical_population.py --screen output\phase_timewarp\population2608_v1 --evaluation output\release_research_20261031\continuation_v1\population_evaluation --output DIRECTORIO_NUEVO
```
