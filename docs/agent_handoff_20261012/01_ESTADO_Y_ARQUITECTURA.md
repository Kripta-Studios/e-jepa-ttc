# 01 — Estado final y arquitectura que se evaluó

## Estado a fecha de corte

Los doce fits V13 están completos. Sus contadores, hashes de checkpoint e
identidades figuran en `evidence/PRODUCERS_PAIR_ENDPOINT_FREEZE.json` y
`evidence/HEADS_V_ROUND_FREEZE.json`; el informe 04 genera la tabla de updates.
También terminaron las características e historias H/V, normalizadores,
predicciones V, evaluación V, inferencia y scoring Dev32, costes y reporte.

La cola histórica conserva `campaign_status=TERMINAL_WITH_BLOCKERS` porque
`FCWD_RGB_STATUS` exige calibración oficial y sigue `BLOCKED_EXTERNAL`. Esto
**no significa que los entrenamientos estén pendientes**. La nueva campaña
FCWD con registro estimado terminó por una ruta distinta, marcada exploratoria.
No se falsificó el estado del contrato original para admitir otra geometría.

La última auditoría de reparaciones externas verificó 1.412 archivos protegidos
sin cambios de SHA256: [FINAL_QA](../fcwd_rgb_registration_20261012/FINAL_QA.json).
Los cero updates de ese recibo corresponden a la intervención de evaluación,
no a toda la campaña de entrenamiento. Las suites de días distintos se solapan;
no sumar sus recuentos como si fueran casos únicos.

## Qué significa cada nombre

- **E**: eventos. **R**: RGB. Son productores separados, con entradas y pesos propios.
- **A5** y **C2F**: variantes del productor geométrico `CausalScaleTTC`; C2F combina
  transporte fino y grueso. La señal aprendida corrige una estimación geométrica.
- **PAIR_E_MATCHED / PAIR_R**: experto de pares dentro de cada modalidad. PAIR no
  significa que E_A5 y R_A5 se mezclen: la fusión multimodal es posterior.
- **E_H1 / R_H1**: corrector sobre una observación; **E_CTX / R_CTX**: corrector
  con contexto temporal. No atribuir a H8 un router de pesos de expertos: es un
  corrector residual; el router C2F tiene otra función.
- **F_TRUE**: fusión con la información multimodal prevista por la receta.
  **F_ZERO**: control de fusión con la información RGB anulada según su contrato.
  Compararlas ayuda a separar información de capacidad, no es un baseline Garl.
- **V12 H8**: contexto de ocho observaciones sobre productores compartidos.
  La ruta histórica completa usa tres cabezas, semillas 7/13/23, y mediana de
  sus TTC. Las filas `H8_seed*` son cabezas individuales y no la mediana.

Consultar las definiciones efectivas en `src/e_jepa_ttc/models/causal_scale_ttc.py`,
`src/e_jepa_ttc/rgb_port/`, `operational/rgb_port/train_producers.py`,
`train_heads.py` y los módulos de `src/e_jepa_ttc/rgb_port/` en la revisión entregada.
Los entrypoints auxiliares se localizan con `rg --files operational/rgb_port`.

## Por qué V13 volvió a entrenar eventos

V13 estudia eventos, RGB y fusión con una división de roles propia: P para
productores/PAIR, H para cabezas y normalización, V para evaluación. El protocolo
usa 24 grupos P, 8 H y 8 V. Reutilizar sin más productores V12 entrenados con
TRAIN40 invalidaría la separación de roles de esta comparación. Tener menos
grupos P cambia exposición e información; por sí solo no prueba ni refuta que
V13 vaya a generalizar peor. Los resultados externos sí muestran que E_CTX V13
no domina a H8 V12.

Los manifests ejecutados del run son la autoridad para población, hashes y
asignación. El número de grupos no equivale al número de muestras independientes:
las ventanas dentro de un episodio están correlacionadas.

## Raíces locales

```text
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/
  e-jepa-ttc-v12-efficient-context/   código y evidencias históricas V12
  e-jepa-ttc-v13-rgb-port/            código y documentación V13
    artifacts/rgb_port_20261008/     run canónico V13

E:/EJEPA_results/fcwd_rgb_calibration_20261012/campaign/
  registration/                     registro RGB por consulta
  predictions/                      predicciones, sellado, scoring
E:/FCWD_dataset/                     medios y calibración FCWD
E:/eAP_dataset/                      medios eAP
E:/GarlTTC_dataset/                  anotaciones y entradas GarlTTC
```

Dentro del run V13: `fits/<FIT>/checkpoint_last.pt`, `rounds/`, `evaluation/`,
`profiles/ROUTE_COSTS.json`, `transfer/dev32/`, `transfer/dev32_score/`,
`report/REPORT.md`. Los snapshots públicos permiten inspeccionar identidades sin
tener los pesos; no permiten ejecutar la red por sí solos.

## Qué no ha terminado científicamente

No hay resultado oficial CodaBench test12. No hay comparación FCWD Garl
multimodal con la nueva transformación. No hay benchmark aislado completo de
latencia V13 streaming equivalente al piloto V12. La atribución del error a
productor, contexto, ROI o normalización sigue pendiente; terminar una cola
no resuelve esas preguntas.
