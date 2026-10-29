# Hash fuerte RGB: diagnóstico CPU SHA-256 frente a BLAKE3-256

Generado por `artifacts/train40_runtime_council_20261028/frame_hash_cpu/write_findings.py`.
Fuente: [recibo original](../../artifacts/train40_runtime_council_20261028/frame_hash_cpu/idx553_v1/summary.json);
[distribuciones, pares, procedencia y SHA de origen](FRAME_HASH_CPU_GENERATED.json).

El caso 553 (`EXTENDED:CPNAO-medium:20200738`) conserva los mismos siete
frames auténticos de 6.912.000 bytes, uint8 1200×1920×3, strides [5760,3,1].
Se preservaron 44 pares de scope:
dos warmups y veinte repeticiones medidas para cada uno de los dos scopes.
Orden de algoritmos y scopes alternado AB/BA; BLAKE3 usa `max_threads=1` y
digest de 32 bytes. Todas las rutas pasan comparación de píxeles/bytes y digest
contra su referencia del mismo algoritmo. SHA y BLAKE3 tienen valores distintos;
el recibo conserva ambos y la SHA del factory original para cada frame.
Los 185 archivos protegidos y el stat RAW permanecen intactos.

| Reloj por frame, ms | n | Media | Mediana | p95 | p99 | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Solo hash, sha256 | 140 | 3.2332 | 3.1932 | 3.5395 | 3.8274 | 3.9609 |
| Solo hash, blake3 | 140 | 0.8291 | 0.8167 | 0.8944 | 1.0169 | 1.0918 |
| Copia + hash, sha256 | 140 | 3.9973 | 3.9831 | 4.2186 | 4.4770 | 4.9172 |
| Copia + hash, blake3 | 140 | 1.6342 | 1.6047 | 1.8056 | 1.9943 | 2.1029 |

Cada distribución usa directamente 140 `frame_checks` medidos por algoritmo y
scope; no divide percentiles del historial entre siete. El JSON conserva además
las distribuciones por frame ID, por historial y de diferencias pareadas.
La diferencia media pareada SHA − BLAKE3 para copia + hash es
**2.3632 ms por frame**. Las medias por historial de siete
frames son 27.9978 ms con SHA y
11.4535 ms con BLAKE3.

## Qué justifica y qué falta

El ahorro medido permite investigar una V2 opt-in que mantenga identidad
criptográfica del contenido, sin sustituirla por un ID de capacidad local.
V2 todavía no está implementada. V1 SHA sigue siendo el default y conserva su
descriptor e identidades. Una implementación debe rotular explícitamente
`blake3-256`, separar el namespace de identidad y usar SHA-256 solamente para
el descriptor de metadata. Nunca debe llamar SHA-256 al digest BLAKE3 ni volver
a calcular una SHA completa oculta al consultar propiedades.

Esto es coste de copia/hash aislado, no un ahorro TTC medido. El factory completo
también valida, construye metadata y publica el token en su registro; esos costes
no están en la tabla. El setup, HDF5, referencias y comprobaciones se excluyen de
los relojes. El scope sin copia parte de bytes existentes; el scope con copia
crea bytes nuevos cada ruta. Ninguna ruta representa llegada causal, colas,
transferencias ni modelos. Un update sin frame nuevo no paga sellado; con bursts
debe medir todas las llegadas, sin prehash de observaciones futuras.

No se puede restar el p95 de aquí al p95 de otro ensayo. Las 140 observaciones
comparten veinte repeticiones y siete frames del mismo historial; no son 140
episodios independientes. El resultado no demuestra 50 Hz ni <20 ms completos,
no cambia pesos/normalizadores y no admite ampliar a dos objetos sin medir TTC.

## Dependencia y reproducción

Se verificó el wheel binario privado `blake3==1.0.11` de
220,544 bytes, sin modificar venv compartido, pyproject ni lockfile.
SHA-256 del wheel: `de3fbfeef38f68b32c23ae954a83bbfc0c69189c480b045f91ae55e0f0ef9007`.
Licencia declarada: `CC0-1.0 OR Apache-2.0`. Véanse
[recibo e instalación privada](../../artifacts/train40_runtime_council_20261028/hash_probe_dependencies/README.md),
[bindings oficiales](https://github.com/oconnor663/blake3-py) y
[release fijada](https://pypi.org/project/blake3/1.0.11/).
El diagnóstico comprueba archivos instalados contra el wheel, origen del módulo
y de su extensión, versión y vector vacío oficial; la referencia incremental
usa la misma implementación, no constituye una segunda implementación independiente.

Host `DESKTOP-2N393IP`, Python 3.11.15, NumPy 2.4.6,
reloj `QueryPerformanceCounter()`, resolución
1.0e-07 s; OMP/MKL=2/2.
El JSON enlazado conserva comando, commit, entorno, referencias y provenance completa.
Regeneración de este documento (stdlib, sin repetir la medición):

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' artifacts/train40_runtime_council_20261028/frame_hash_cpu/write_findings.py
```
