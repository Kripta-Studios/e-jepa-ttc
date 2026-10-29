# Viabilidad de eventos eAP para la fase S4

Fecha de auditoría: 2026-10-29. Resultado: **la rama de eventos 50/100 ms con
cinco bins es viable con lectura indexada y acumulación por chunks**. La
causalidad que puede acreditarse es sensorial respecto al final de exposición
RGB; el release no aporta timestamps de llegada o de paquetes.

El recibo normativo es
`output/event_feasibility/eap_train_event_feasibility_v4.json`, SHA-256
`21e3726c1da791957176bb58b997825633ee4a158e169de0db369a52cf339eea`.
Registra el script ejecutado con SHA-256
`c3846d5d498fbfabf9bc199bfc8f7bbb56be4ec7aa1c56c5828acbba44a17279`.
La primera salida sin sufijo usó una comprobación cerrada del último
microsegundo; `_v2` todavía no registraba el hash del script y `_v3` no incluía
el resumen global de calibración. Esas salidas preliminares no se conservaron;
`_v4` es el único recibo admisible.

## Contrato comprobado

La revisión local fijada usa `data/train.parquet` con SHA-256
`69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314`.
Contiene 118.247 frames de 40 secuencias y las columnas de inicio/fin de
exposición, `K_event`, `T_event_ego` y ruta HDF5. No contiene timestamp de
llegada, timestamp de paquete, pose temporal del vehículo ni una columna de
resolución.

Los 40 HDF5 train ocupan 448.999.857.897 bytes y declaran 179.449.844.003
eventos. Cada stream contiene:

| Dataset | Tipo | Interpretación comprobada |
|---|---|---|
| `events/x` | `uint16` | coordenada en el plano de eventos |
| `events/y` | `uint16` | coordenada en el plano de eventos |
| `events/t` | `int64` | reloj entero compartido con las exposiciones RGB |
| `events/p` | `int8` | valores raw 0/1; el código los normaliza a -1/+1 |
| `ms_to_idx` | `uint64` | índice por milisegundo absoluto, consultado con `t_us // 1000` |

Los tres probes, 200.000 eventos por secuencia, observaron `x=0..1279`,
`y=0..719`, ambas polaridades y timestamps no decrecientes dentro de cada chunk.
Esto coincide con `EAP_IMAGE_SIZE=(1280,720)`. Es evidencia acotada, no un scan
de cada coordenada o timestamp de los 179.450 millones de eventos.

Treinta y ocho streams usan el filtro HDF5 Blosc2 y exigen importar
`hdf5plugin`; `q2k6O0X6JY` y `rhYxPcU29n` están sin filtro. Los cuatro arrays de
eventos tienen la misma longitud en cada secuencia auditada. No hay atributos de
unidad en HDF5: la unidad microsegundo se establece por la coincidencia con las
columnas `_us`, el índice por milisegundo y el lector versionado del repositorio.

`K_event` y `T_event_ego` tienen un único valor en las 118.247 filas; sus hashes
float64 little-endian son, respectivamente,
`2e293630fb55ed09449333c7040b7c634d061e8ad06345f8a6086cab60558d32` y
`8ecac5d2c61d39fe1450de58564cdb35cdcc4ab538422fa2296d4dd8bc218bb2`.
La matriz completa queda en cada probe del recibo. `T_event_ego` es una
extrínseca estática; no autoriza compensación temporal por ego-motion.

## Ventana causal propuesta

Para una decisión emitida después de terminar la exposición RGB:

```text
cutoff_us = rgb_exposure_end_timestamp_us
events = [cutoff_us - window_us, cutoff_us)
window_us ∈ {50_000, 100_000}
```

El extremo derecho exclusivo impide consumir eventos posteriores al fin de
exposición. Las 40 secuencias tienen endpoints que cubren al menos 100 ms antes
del primer fin de exposición y cubren el último intervalo semiabierto. Esto no
demuestra tiempos de llegada ni ausencia de huecos internos; por tanto, el claim
admisible es causalidad en reloj sensorial.

Los tres frames centrales auditados produjeron:

| Secuencia | Eventos 50 ms | Eventos 100 ms |
|---|---:|---:|
| `2Y46OkK3ff` | 1.752.588 | 3.468.742 |
| `Po6ePjWYAz` | 430.701 | 836.430 |
| `qGsgzl4Q8B` | 2.014.179 | 3.962.377 |

Estas densidades descartan cargar la ventana completa como estrategia general.
También justifican registrar tasa, clipping y porcentaje de bins saturados en
cada experimento.

El contrato inicial S4 debe usar cinco bins por polaridad, diez canales, con
interpolación temporal lineal y normalización por ventana. La acumulación se
hace directamente desde chunks; eventos raw 0 pasan al canal negativo y raw 1
al positivo. El control de 100 ms cambia solo la duración, no el número de bins.

El dataset SSL existente usa timestamps de centro de exposición y construye
targets futuros. Ese comportamiento es válido para su objetivo predictivo, pero
no debe reutilizarse como input del detector causal. La rama S4 necesita el
cutoff de fin de exposición anterior y no recibe ventanas futuras.

## Memoria e I/O

El lector existente `EAPEventReader.iter_window_chunks()` conserva el intervalo
exacto y limita cada lectura a 250.000 eventos. Los arrays tipados `x/y/t/p`
representan 17 bytes por evento tras conversión, 4,25 MB nominales por chunk;
las máscaras y copias de selección elevan el pico real y deben perfilarse.

Un acumulador FP32 de cinco bins por polaridad requiere:

| Resolución | 10 canales | 12 canales compatibles con count/rate |
|---|---:|---:|
| 160×90 | 0,576 MB | 0,691 MB |
| 640×384 | 9,830 MB | 11,796 MB |
| 1280×720 | 36,864 MB | 44,237 MB |

La primera implementación del detector debe acumular a 160×90 o a la resolución
de feature elegida, mantener como máximo dos handles HDF5 por worker y usar un
LRU pequeño de voxels. No debe compartir handles entre procesos, cachear streams
completos ni materializar full-resolution voxels para todo el batch. En Windows,
se comienza con `num_workers=0/1` y se aumenta solo después de medir RAM,
throughput y presión de página. La configuración SSL actual mantiene un reader
por secuencia y ocho workers; no es el contrato de memoria aprobado para S4.

## Gates antes de entrenar F01/F11

1. Fixture que rechace cualquier evento con `t >= cutoff_us` y un control que
   demuestre que añadir eventos futuros falla.
2. Paridad entre voxelización de referencia y acumulación chunked para 50/100 ms.
3. Transformación espacial única y registrada entre el plano 1280×720, el resize
   RGB y las features. El RGB publicado está remapeado aproximadamente al plano
   de eventos; no se presupone alineamiento subpíxel.
4. Métricas de eventos por ventana, chunks, tiempo de lectura, RAM y handles.
5. Normalizadores ajustados solo en train y roles de secuencia conservados.
6. Comparaciones F00/F10/F01/F11, eventos cero/permutados/desfasados y control
   RGB de capacidad comparable bajo el mismo presupuesto.

## Reproducción

```powershell
$python = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
$env:PYTHONPATH = 'src'
& $python scripts/audit_eap_event_feasibility.py `
  --root E:/eAP_dataset `
  --output output/event_feasibility/eap_train_event_feasibility_REPRO.json `
  --sequences qGsgzl4Q8B 2Y46OkK3ff Po6ePjWYAz
Get-FileHash output/event_feasibility/eap_train_event_feasibility_REPRO.json -Algorithm SHA256
```

El timestamp de creación hace que el hash de una reproducción nueva cambie. Los
conteos, endpoints, matrices, hashes de metadata y probes deben coincidir. El
script no abre labels, test, RGB ni GPU y nunca lee un stream completo.
