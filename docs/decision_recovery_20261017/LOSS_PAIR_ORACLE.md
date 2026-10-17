# Oráculo P de la pantalla L0/LW

Diagnóstico no desplegable: se escoge el menor MiD por consulta usando GT, con empate a L0. Todas las métricas usan ese mismo vector. No se abre H ni se ajusta un router.

| Modelo | MiD agregado | Crítico | Pequeño | Largo | Negativo | MAE s | Signo |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 59.652347 | 59.786080 | 49.213810 | 56.404888 | 93.546748 | 5.926142 | 6.4131% |
| LW | 60.027751 | 63.687802 | 51.302320 | 56.149822 | 71.781713 | 4.784872 | 4.3626% |
| A5 | 100.873656 | 141.812711 | 62.952010 | 55.295382 | 55.521592 | 4.750671 | 3.3327% |
| ORACLE | 49.745959 | 51.159725 | 42.021706 | 47.568692 | 68.027160 | 4.328413 | 3.9840% |

Selección: 21,867 L0 y 32,023 LW de 53,890 consultas P. A5 es referencia, no un experto seleccionable.

Frente a L0: cambio agregado -16.607%; rescates/daños de signo 1309/0; mejoras/daños de error absoluto 29876/2147. Gates numéricos fallidos: [].

Frente a LW: cambio agregado -17.128%; rescates/daños de signo 204/0; mejoras/daños de error absoluto 21167/512. Gates numéricos fallidos: [].

Frente a A5: cambio agregado -50.685%; rescates/daños de signo 1193/1544; mejoras/daños de error absoluto 31100/22730. Gates numéricos fallidos: ['negative', 'sign'].

El mínimo MiD por consulta es también el mínimo posible de cada banda para cualquier selector duro entre estos mismos outputs L0/LW, porque todos los pesos registrados son positivos. Su MiDn supera el límite original frente a A5; ningún selector puro de este par puede cumplir esa protección en este P, aunque sí hay complementariedad. No es un límite para interpolar, cambiar modelos o evaluar otras poblaciones.

La complementariedad mide margen de selección ideal; no demuestra que un gate causal pueda aprenderlo. Es P de entrenamiento, una semilla y padres compartidos. No habilita promoción, nueva réplica ni una conclusión sobre generalización.

Las saturaciones, rescates/daños por banda y grupo están en REPORT.json. El Parquet P_ORACLE.parquet conserva IDs, targets, relojes originales, experto escogido y vector único. RECEIPT.json vincula entradas, código y salidas. Cero updates y cero inferencias.
