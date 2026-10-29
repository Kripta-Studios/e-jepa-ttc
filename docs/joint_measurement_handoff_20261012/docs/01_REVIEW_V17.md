# Revisión independiente de V17

## 1. Alcance material

Fuente: release `v17-research-results-20261012`, commit `396abe922537d9dac19d6db3451e59d283636e20`.
ZIP recibido: 365.581.652 bytes; SHA-256 `8ce10a722967e6f5c49e13fe176c494d562678e35b3f6c834d7d38fd005e6007`.
Se comprobaron CRC, rutas, duplicados y 5.211 hashes internos (5.212 miembros). Git clone HTTPS falló por DNS; no se afirma haber obtenido .git. Se inspeccionó source/ del ZIP y el conector GitHub.

Los verificadores oficiales estrictos de métricas/exports no pasaron igualdad bit a bit en Linux/Python 3.13.5/Torch 2.10.0+cpu/NumPy 2.3.5/Pandas 2.2.3. Se preserva ese fallo. El recálculo numérico de 44 tablas y 1.360.146 filas dio máximo delta 1,3194e-5 entre escalares métricos, sin diferencias estructurales/no numéricas. Los 23 exports ejecutados sobre ejemplos incluidos cumplen una comparación DESCRIPTIVA atol=rtol=1e-5; máximo delta absoluto 7,6294e-5 en campos de distinta escala. No se amplía una tolerancia científica de campaña ni se sustituye su recibo. No hay nuevas inferencias de todas las filas o entrenamiento.

## 2. Resultado global

23 fits; 106.500 updates científicos y 180 técnicos. 28 ajustes cerrados también son entrenamiento. 10.655,864 s dentro de scopes GPU, no tiempo puro de kernels. A/B/D/E/F/G/H/I terminan NO_CANDIDATE; C diagnóstico completo. Todos comparten representación/encoder congelado. No extrapolar el millón autorizado como millón ejecutado ni tomar su saldo como permiso nuevo.

### Resumen interpretativo

| Cola | Resultado | Inferencia permitida |
|---|---|---|
| A: 896 tokens en H8 | MiDc V: Z=157,089, G=148,355 en media de tres seeds; overall 106,545→102,762; MAE 2,580→2,653 | Hay señal útil parcial, con mayor ganancia de ajuste H que de V. No cumple el criterio registrado. |
| B: precisión NLL constante | MiDc H mejora alrededor de 1,5–1,9%; no 10% | La ponderación influye; no es suficiente bajo esa receta. |
| C: ridge dejando grupos H fuera | RD MAE fase 0,00650058 vs R0 0,00678073 (~4,1%) | Pequeña señal lineal fuera de grupo; no prueba un H8 mejor. |
| D: diferencias de tokens | V MiDc ~148–150, no mejora consistentemente a G | Quitar nivel estático/reducir dimensión no resolvió el compromiso. |
| E: altura lineal | Una predicción H y cinco P con MiD inválido; fuertes regresiones | Ajustar alturas aisladas no garantiza TTC válido/preciso. |
| F: altura + cambios | H overall 135,556; conserva cobertura pero falla calidad | Supervisar diferencias en ese readout no basta. |
| G: pooling vertical | H overall SP0=112,342, SP1=138,727 | No funcionó esa lectura de altura; no es un test de correlación espacial de pares. |
| H: sin clipping | H overall 108,780 vs 108,569 control | El clipping no era la explicación suficiente propuesta. |
| I: MLP de altura | H overall 160,671 vs 138,727 control | Más capacidad en esa calibración empeora. |

Las métricas de productores H, cabezas V, TRAIN40 y FCWD no son intercambiables. No existe en V17 una nueva comparación FCWD contra Garl.

## 3. Geometría: qué sí cuantifica el diagnóstico

Se recalcularon 53.890 filas P (ver `evidence/GEOMETRY_POPULATION_RECALC.csv`). Para 208 TTC≤1 s, mediana de log-ratio requerido=0,108897; caja anotada=0,104967; altura aprendida=0,008431. En toda P, desviación de altura aprendida≈0,005153 vs anotada≈0,090569. El cociente de dispersiones ronda 0,057. Es escasa variación respecto al target, no un diagnóstico de colapso de TODO el encoder.

Las cajas son targets/diagnóstico TRAIN, no inputs ocultos para inferencia. Su ratio no es un oráculo perfecto: en 6–10 s su MAE de ratio es 0,015596, mayor que 0,012711 de la geometría aprendida. Las aproximaciones físicas, oclusiones y ruido de etiquetas permanecen.

Una mejora de altura absoluta puede empeorar el cambio temporal. Si log-altura estimada=u_t+e_t, el error del ratio es e_t−e_{t−1}. Bajo iguales varianzas, Var(e_t−e_{t−1})=2σ²(1−ρ). Bajar σ sin conservar correlación temporal puede aumentar el error de diferencia. Esto es un mecanismo matemático posible, NO una estimación causal de ρ en esta campaña.

El piloto foreground tiene gradientes finitos/no nulos en 40/40 consultas. No prueba que esa rama esté muerta ni permite sumar o comparar normas de pérdidas con escalas/reducciones distintas como porcentajes de responsabilidad. Las calibraciones E/F/G/I no entrenaron ese foreground.

## 4. Decisión

No repetir más calibraciones ridge/MLP sobre los mismos vectores ni volver a apagar clipping o homogeneizar NLL. No promover G porque tenga mejora parcial. Tampoco convertir el rechazo de esos componentes en prohibición de entrenar una representación nueva.

La siguiente hipótesis es aprender una MEDICIÓN RELATIVA desde dos mapas espaciales conjuntamente, supervisada por TTC y con gradiente hasta la CNN. Un control sin correlación separa el efecto de esa estructura del nuevo encoder/readout. Un control correlación+encoder congelado separa usar un mapa existente de adaptar la representación. El entrenamiento end-to-end se ejecuta incluso si el control congelado pierde.

No se afirma que esta arquitectura sea nueva en toda la literatura ni que vaya a superar Garl. Debe compararse con Stage70/PPS o cualquier implementación local previa equivalente antes de duplicar trabajo; código/config sin endpoint no es prueba de ejecución.

## Fuentes

- source/docs/representation_audit_20261012/{RESULTS,QUEUES_RESULTS,MECHANISMS,FOREGROUND_PILOT,CAMPAIGN_STATUS}.md
- campaign/diagnostics/P_GEOMETRY_CONSISTENCY_ROWS.csv
- source/src/e_jepa_ttc/models/causal_scale_ttc.py
- release pública V17 y los protocolos adjuntos del usuario.
