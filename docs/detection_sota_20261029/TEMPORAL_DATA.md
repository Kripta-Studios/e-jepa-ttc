# Fundamento de datos RGB temporal causal

Fecha: 2026-10-29. Este contrato prepara S3 sin introducir todavía un modelo,
configuración de entrenamiento ni nueva afirmación de precisión.

## Relojes disponibles

El `data/train.parquet` fijado publica por frame:

- `rgb_exposure_start_timestamp_us`;
- `rgb_exposure_end_timestamp_us`.

No publica timestamp de llegada, fin de decodificación ni disponibilidad para el
proceso. Por ello el índice acredita **causalidad en tiempo sensorial**. No acredita
latencia física de adquisición. La decisión sensorial del frame actual se fija en su
fin de exposición; un frame pasado solo se admite si terminó como máximo al comenzar
la exposición actual:

```text
past.exposure_end_us <= current.exposure_start_us
decision_sensor_us = current.exposure_end_us
```

La igualdad en el cutoff está permitida. Un frame solapado, futuro, desconocido o de
otra secuencia se rechaza o queda fuera de la historia.

## Contrato

`TemporalRGBIndex` valida identidad, intervalos, timestamps estrictamente crecientes
dentro de cada secuencia y tokens únicos. `resolve()` devuelve el frame actual y dos
slots de historia con:

- tokens y fin de exposición;
- edad respecto al fin de exposición actual;
- máscara explícita por slot;
- control aplicado;
- alcance declarado del reloj.

El inicio de una secuencia no se rellena con otra secuencia. Se representa con slots
`None` y máscara falsa. El segundo frame puede tener un único pasado válido. Esto
permite entrenar o abstenerse sin fabricar historia.

## Controles preregistrables

| Control | Contenido de los dos slots |
|---|---|
| `current_only` | Ambos ausentes; baseline de un frame |
| `current_repeated` | El frame actual repetido; misma capacidad sin información pasada |
| `correct_past` | Los dos frames pasados causales más recientes |
| `delayed_past` | Omite el pasado más reciente antes de elegir dos |
| `permuted_past` | Dos frames pasados causales de la misma secuencia elegidos por hash fijado |

`permuted_past` nunca usa futuro ni cruza secuencias, pero puede producir historia muy
antigua y fuera de distribución. Debe interpretarse junto con controles entrenados de
capacidad, no como prueba aislada. `current_repeated` es el control limpio de capacidad.

## Auditoría del release local

Se leyeron solo las cuatro columnas de identidad/exposición del Parquet canónico, sin
abrir RGB, eventos, etiquetas ni test. Se conservaron los roles 24 train y ocho
selection del split candidato.

La fuente numérica es el recibo regenerable
`output/campaign_data/temporal_audit_v2/audit.json`, SHA-256
`99ea4af1b6f83d2ce7aa5b646059fecb9d8086c9f3a4307e4d45b1301edb5f53`.
El recibo fija el Parquet
`69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314`,
el split `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903`,
el módulo `0ddea06600111cf359145ead1a107c28cb59117c56f677e792650758accc2c3d`
y el entrypoint `cab8d3a21b0c500f16db9f68438bfb494e49f34693213ed262bb3440f6fd6648`.
La tabla siguiente reproduce exactamente `support.roles` de ese JSON.

| Rol | Secuencias | Frames | Sin historia | Un pasado | Dos pasados |
|---|---:|---:|---:|---:|---:|
| train | 24 | 70.282 | 24 | 24 | 70.234 |
| selection | 8 | 23.958 | 8 | 8 | 23.942 |

Las edades observadas de los dos slots van de 96.612 a 400.005 µs en train y de
98.760 a 399.879 µs en selection. Por tanto, describir la historia simplemente como
0,1–0,2 s no cubre todos los huecos del release. El modelo futuro debe recibir edades
reales y máscaras; no debe asumir frecuencia perfectamente uniforme.

## Gates antes de S3

1. Fijar manifiesto D4096, selección y seed antes de producir historias.
2. Registrar la política de frames faltantes y edades, no interpolar silenciosamente.
3. Comparar un frame, actual repetido, pasado correcto, pasado retrasado y pasado
   permutado con igual población y presupuesto.
4. Separar causalidad sensorial de latencia de llegada, que sigue desconocida.
5. No usar coordenadas ego entre frames como marco estacionario sin ego-pose.
6. Conservar el reset por secuencia en train, validación, resume e inferencia.

## Límites

- No se verificaron timestamps de llegada porque no existen en el manifiesto.
- No hay agrupación conocida de adquisiciones relacionadas entre `sequence_id`.
- La historia RGB no demuestra identidad de objeto ni tracking.
- El índice no ejecuta un detector temporal. La integración descrita en
  [TEMPORAL_MODEL.md](TEMPORAL_MODEL.md) necesita paridad real y entrenamiento.
- Un beneficio futuro debe superar `current_repeated` y el baseline de un frame.
