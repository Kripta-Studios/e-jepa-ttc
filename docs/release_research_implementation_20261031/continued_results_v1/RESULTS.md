# Investigación continuada del release

Los contrastes nuevos están ejecutados. No acreditan una mejora del release;
localizan fallos de cobertura, geometría y paridad que condicionan el siguiente paso.

## Observabilidad RAW O0

149 consultas; 148 pares evaluables.
Las seis variables físicas se extrajeron antes de unir targets.
Se hicieron 100 ajustes ridge con adquisición excluida y cuatro brazos.
El escalado y la imputación se ajustan solo en entrenamiento.
Los padres R0 están expuestos: no es OOF de la cadena ni confirmación independiente.

| Sonda de utilidad de la corrección | MSE macro |
|---|---:|
| constant | 0.457492901 |
| permuted | 0.756780224 |
| physical | 0.919267453 |
| support | 0.492043720 |

Delta físico−soporte: 0.427223732; CI95 por adquisición [0.08055069039001245, 1.010417232067282].
Admisión del gate por la regla prefijada: **False**.
MSE corresponde a predecir gain, no al error TTC del release.

| Indicador físico | Pares con medición |
|---|---:|
| foreground_background_disagreement | 7 |
| independent_patch_fraction | 120 |
| normalized_residual | 120 |
| null_leakage | 120 |
| q_sensitivity | 63 |
| temporal_innovation | 40 |

## Coordenadas y superficies V0/V1

Medidas RGB representables: 133/149.
MAE macro R0: **0.766024827 s**;
superficie con fallback R0: **3.103982972 s**.
CI95 del delta por adquisición: [0.9342313089663523, 4.198244943792357].
Fallback en 16 consultas; ausencia original conservada.
Cambios de crop dentro del triplete: 0;
entre snapshots: 0; resize anisótropo: 53.
Coordenadas auditables en 146/149 consultas.
El resize es constante en el triplete; no aparece zoom artificial variable.
Esto no descarta errores de oclusión, fondo, paralaje o identidad de superficie.
No se admite V2/V3: la correspondencia consistente por sí sola no garantiza TTC.

## Preparación del auxiliar de sensor

Se inspeccionó la calibración de 25 adquisiciones.
| Familia de campos | Adquisiciones con candidatos por nombre |
|---|---:|
| contrast_or_threshold | 0 |
| exposure | 24 |
| gain | 0 |
| light_reference | 0 |
| refractory | 0 |
| response | 0 |

La calibración geométrica y exposición RGB no identifican la escala fotométrica.
Faltan referencia anclada, escala identificada y separar el error de alineación.
E_C2F expone mapas densos mediante una opción de forward; PAIR133 no los conserva.
S1 real y S2/S3 requieren resolver ese dato; no se inventa una calibración.

## Captura de despliegue y paridad

Se capturaron dos observaciones de cada modalidad desde medios verificados.
El replay local de PAIR/PHASE17 es exacto. La comparación con el archivo histórico da
**FAIL** en 2/2 consultas.
El fallo afecta a tensores intermedios. TTC pasa en 2/2.
No se ejecutó el candidato. El siguiente control fijado repite el original en CUDA
con las mismas tolerancias cuando termine la población y F0 quede admitido.
Se conservan dos intentos técnicos fallidos: clave del inventario y contrato antiguo
de máscara; el intento válido usa NativeObservation, como el despliegue archivado.

## Continuación y límites

El worker population2608 sigue siendo la única extracción ampliada. La continuación
espera su COMPLETE, verifica fuentes, evalúa F0 y publica el informe. Solo si F0 pasa
ejecuta el perfil CUDA; solo si su paridad pasa abre la captura nativa completa y
exige nuevamente paridad original antes de la intervención canónica.
O1/O2 y V2/V3 no quedan admitidos por estos resultados. D4 y JEPA tampoco se abren
sin sus evidencias específicas. Ningún componente o normalizador de R0
se ha actualizado. La campaña no está completa: siguen pendientes cadencia ampliada,
conservación nativa, confirmación independiente y latencia completa del candidato.

## Regeneración

```powershell
python scripts/report_release_research_continuation.py --output DIRECTORIO_NUEVO
```
