# Wiring V2: propuesta pendiente de implementación

El core de identidad V2 está implementado y validado en CPU en `9e752ae4`;
véase [estado y contrato](FROZEN_RGB_V2_DESIGN.md). Este documento propone
conectar ese core a fuentes causales y al runner. **No se ha editado runtime
para este wiring.** Al cierre solicitado por el usuario se congela también la
investigación: este documento queda como handoff, sin implementación programada.
El default sigue siendo SHA-256 V1. El core tiene 101 tests CPU aprobados y un
rerun de 20 tests V2; esa QA no prueba latencia TTC.

El crop companion V1 pasó 18/18 actualizaciones con paridad y 623 archivos
preservados. El [control paced memory/crops/reverse de 250 consultas](../../artifacts/train40_runtime_council_20261028/paced/memory_crops_reverse1_250_v1/summary.json)
terminó FAILED por dos drops y bloqueo posterior de historia: 115/250 salidas,
todas con paridad, 724 archivos preservados. No hay cobertura completa ni mejora
admitida de 50 Hz/20 ms; no se ha atribuido causalmente el fallo a la caché.
La ventaja CPU BLAKE3 sigue siendo sólo de un micro; V2 no está conectada a
fuentes, runner o TTC. Revisión de alcance coordinada con `council_rgb`; no se
ejecutaron QA ni mediciones para preparar este diseño.

## Selección explícita y combinaciones admitidas

Propuesta de flags en `scripts/benchmark_train40_paced.py`:

```text
--rgb-arrival-hash {sha256,blake3-256}       default: sha256
--rgb-hash-dependency-path PATH            default: None
```

| Fuente | SHA default | BLAKE3-256 opt-in |
|---|---|---|
| `disk` | Control actual de arrays readonly, sin convertirlo a fuente sellada | Rechazar antes de recursos |
| `persistent` | Control actual de arrays readonly/copias, sin convertirlo a fuente sellada | Rechazar antes de recursos |
| `immutable` | Fábrica V1 actual | Backend verificado en `ImmutablePersistentRawSource` |
| `memory` | Fábrica V1 actual | Backend verificado en `PreloadedRawSource` |

Un flag BLAKE3 exige ruta privada. Una ruta privada con SHA se rechaza para evitar
una configuración ignorada. No se infiere BLAKE3 por tener un paquete instalado,
ni se cambia de algoritmo en fallback. `sha256` en los controles legacy conserva
su comportamiento previo y el binding `OWNED_READONLY_PAYLOAD_ARRAYS`; no afirma
que ya exista un token de llegada en esos controles.

Un helper de validación se ejecutará al inicio de `run`, antes de abrir HDF5,
preload, readers/pools, modelos o llamadas CUDA. La ruta de dependencia se
añadirá a los directorios protegidos de `_validate_output`; output nunca podrá
pisar su instalación. Argumentos incompatibles deben fallar antes de invocar
loader, verificación de candidatos o constructores de fuente. El parsing de
`--help` seguirá sin inicializar GPU.

## Propagación mínima a fuentes

Se añadirá `hash_backend: VerifiedBlake3 | None = None`, keyword-only, a:

- `ImmutablePersistentRawSource.__init__` en el runner. Su
  `_prepare_frames` pasará el backend a la llamada existente de `freeze_rgb_frame`.
- `PreloadedRawSource.__init__` en `replay_memory.py`. Su `_release_frames`
  pasará el mismo backend a la llamada existente de fábrica.

Los constructores autenticarán el backend sin leer píxeles. La selección queda
fijada durante todo el lifetime de la fuente; no habrá setter, callback de
hashing, cambio por update ni parámetro que contradiga al backend real.
`PersistentRawSource` seguirá siendo el control legacy y padre de la fuente
inmutable; no necesita aceptar un backend que nunca utiliza. El runner pasará
el kwarg sólo a las dos fuentes que sellan, sin reinterpretar otra fuente.

En ambos caminos se conservarán exactamente el cutoff de warmup
`first_anchor - 20_000 us`, capacidad 16, ventana causal, IDs, clocks y bytes.
Se congela cada frame nuevo cuando llega a `acquire_raw`, incluidos frames no
usados todavía por un triplet. Ningún token futuro ni hash de píxeles se prepara
en constructor, `preload` o setup del backend. `memory` puede guardar RAW futuro
privado como ya declara su scope; no puede sellarlo antes de su timestamp.

Los eventos, anotaciones, calibración, geometría, triplets, modelos, pesos y
normalizadores no cambian. Se mantienen los límites de memoria y el mismo
payload conservador, aunque distintos updates compartan tokens. Los bytes de la
extensión nativa/overhead del backend se declaran como exclusión de los límites
numéricos, separadamente del ring y del scratch RAW ya contabilizados.

## Setup, binding y preservación

Orden propuesto para el runner:

1. Validar flags/source y proteger output/dependencia.
2. Si procede, `load_verified_blake3(private_path)` una vez; guardar tiempo de
   setup y recibo. Este paso no recibe frames y precede a CUDA/modelos/preload.
3. Verificar candidato/oracle/manifests, crear schedule y fuente con ese backend.
4. Construir inventario de código y snapshot original, incluyendo
   `backend.protected_paths` explícitamente. Mantenerlos hasta el snapshot final.
5. Ejecutar preload/setup actual, cargar los mismos modelos/normalizadores y
   comenzar el replay sin alterar su reloj ni scheduler.

Las dependencias no deben añadirse a `sources` con `relative_to(ROOT)`: una ruta
privada válida puede estar fuera del checkout. Irán a un bloque separado de
`binding.json` y al inventario de preservación. La copia de código fuente como
`.txt` permanece para fuentes propias; no se copiará un wheel/pyd como texto.
El recibo incluye versión, licencia, pin del wheel, archivos instalados, origen
del módulo/extensión, `max_threads=1`, `digest_bytes=32`, paths protegidos y
duración de verificación. No se instala nada durante el runner.

`rgb_arrival_identity` dejará de afirmar SHA fijo cuando se seleccione V2.
Se preserva el valor V1 actual y se añade un contrato explícito:

```json
{
  "token_version": 2,
  "content_algorithm": "blake3-256",
  "content_digest_bytes": 32,
  "frame_identity_algorithm": "sha256",
  "frame_identity_namespace": "IMMUTABLE_RGB_FRAME_V2",
  "history_identity_namespace": "IMMUTABLE_OWNED_RGB_V2",
  "payload_hash_inside_acquisition": true,
  "future_tokens_or_hashes_prepared": false,
  "max_threads": 1
}
```

Para V1 se declara su perfil real y no se reescribe el descriptor histórico.
`cache_identity` incorporará el contrato y los pins/archivos de dependencia al
hash de runtime. Excluirá del hash los tiempos de setup y otros valores volátiles;
el recibo completo seguirá preservado en binding. El modelo SHA se seguirá
derivando del mismo candidate intacto. Las claves de historia y crop ya separan
versiones mediante sus identidades; no hay que cambiar su aritmética ni API.

## Recibos de llegada y relojes

Stats, warmup, diagnostics de cada payload y lifecycle final declararán perfil
real de sellado, algoritmo de metadata, bytes RAW hasheados y cero copias de
payload compartido. Los contadores actuales conservan significado. Se añadirá
`new_frame_freeze_bytes` como diferencia del acumulado y una lista acotada de
IDs/timestamps nuevos por update, incluidos bursts; cada timestamp debe ser
`<= query_us`. Warmup se identifica aparte por su cutoff y no se presenta como
trabajo de la primera consulta medida.

`new_frame_freeze_seconds` sigue siendo el intervalo interno de la fábrica que
ya existe, no el deadline completo ni una estimación de SHA aislada. La frontera
externa de `acquire_raw` incluye validación, copia, BLAKE3, metadata, publicación
del token y construcción del payload. Ninguna diferencia de un micro se resta
del reloj end-to-end; todos los TTC deben estar disponibles antes de cerrarlo.

## Paridad y pruebas antes de admitir rendimiento

QA CPU del wiring, con fixtures HDF5 reales pequeñas, cubrirá para **ambas fuentes**:

- V1/default y V2 producen los mismos bytes RGB, dtype, shape, strides, IDs,
  timestamps, triplets, deltas, eventos, calibración, boxes y byte_count.
- `preload` genera cero tokens/hashes; warmup sólo sella hasta su cutoff; cada
  adquisición sella todo y sólo RAW nuevo, exactamente una vez y con un hilo.
- Misma frontera de fallos, bounds, rollback, lifetime de payloads tras eviction
  o close y limpieza por excepción; se mantienen los controles disk/persistent.
- Opciones incompatibles y dependencias ausentes/alteradas fallan antes de
  recursos. El snapshot cubre paths fuera de ROOT sin intentar relativizarlos.
- Receipts coinciden con el perfil real; sin acceso a `content_sha256` en V2 ni
  SHA RAW oculta en los caminos medidos; cambio de perfil no cambia candidate.

Antes de medir V2 sobre el manifest real se requiere evidencia separada de
**bytes de Source V1 frente a Source V2**, no sólo identidades distintas. Debe
usar el mismo schedule completo y comparar los píxeles realmente entregados,
sin introducir esa segunda ruta dentro del replay medido. Se propone un
diagnóstico CPU dedicado que avance las dos fuentes por el mismo índice,
compruebe bytes/metadata directamente y suelte cada par tras compararlo. Su
preflight contará ambas fuentes, archives/rings y ambos payloads; si no cabe,
fallará antes de preload. No duplicará archivos RAW ni alterará datos originales.
Los recibos guardarán hashes propios con su algoritmo, resultados de comparación,
cobertura completa y preservación; no exigirán igualdad SHA/BLAKE3.
Las exclusiones y fallos de cada índice permanecerán explícitos; no se eliminarán
slots para aumentar la cobertura ni se continuará con datos stale.

Después se ejecutará V2 con un objeto y las mismas opciones de crop/transport/
eventos que su control V1. `--verify-parity` debe conservar el oracle original y
comparar todos los TTC/ROIs/queries del schedule, con el mismo candidato y
normalizadores; bytes exactos de source no sustituyen este gate numérico.
Primero habrá piloto diagnóstico y paridad completa; sólo entonces medición
pareada de un objeto sin procesos concurrentes, percentiles/cola/cobertura,
contando todo RAW nuevo dentro del deadline. No se amplía a dos objetos por el
resultado del micro ni por la aprobación CPU del core.

## Companion y reparto de archivos

`council_rgb` identificó un consumidor adicional: el helper
`immutable_history_for_admission` de `benchmark_train40_multiobject.py` serializa
`content_sha256`. Un futuro companion V2 necesitará aceptar el backend y usar
`content_digest`, `content_algorithm`, `token_version` e `identity_sha256` como
SHA de metadata, sin recalcular SHA RAW. V1 debe conservar sus receipts actuales.
Ese ajuste es independiente del wiring paced y queda sujeto a asignación del
propietario; permanece pendiente al cierre.

Implementación mínima propuesta tras autorización:

| Archivo | Responsabilidad |
|---|---|
| `operational/train40_runtime/replay_memory.py` | Backend fijo, propagación al sellado causal y receipts por llegada |
| `scripts/benchmark_train40_paced.py` | Flags/preflight, backend setup, immutable source, binding/snapshots/cache identity |
| Tests nuevos del wiring + tests existentes de ambos source/runner | Compatibilidad default, bytes, causalidad, fallos antes de recursos y preservación |
| Diagnóstico CPU de paridad de fuentes, archivo nuevo a acordar | Evidencia real V1/V2 fuera del replay y antes de medir rendimiento |
| Companion multiobject, sólo en lane RGB y autorización posterior | Admisión de píxeles/deltas/TTC V2 y receipts con algoritmo explícito |

No se propone editar `operational/train40_runtime/paced.py`, scheduler, core V2,
modelos, normalizadores ni cachés. Los cambios permanecen pendientes de una
reanudación y asignación explícita; el cierre no autoriza continuar estos ensayos.
