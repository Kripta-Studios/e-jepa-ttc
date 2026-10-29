# FrozenRGBFrame V2 con BLAKE3-256: core implementado

Estado: **core implementado en `9e752ae4`**, revisión independiente `ship`.
Pasan 101 tests CPU de V2/ring/binder/historia inmutable y una repetición de los
20 tests V2; Ruff, formato y Pyright correctos. Los cuatro archivos implementados
son los de la tabla inferior. Esta validación no ejecutó GPU ni midió TTC.

Al cierre solicitado por el usuario, código e investigación quedan congelados.
**SHA-256 V1 sigue siendo el default.** El
[crop companion V1](../../artifacts/train40_runtime_council_20261028/multiobject/crop_rgb_companion_v1/summary.json)
pasó 18/18 actualizaciones con paridad y 623 archivos preservados. El posterior
[control paced memory/crops/reverse de 250 consultas](../../artifacts/train40_runtime_council_20261028/paced/memory_crops_reverse1_250_v1/summary.json)
terminó **FAILED**: dos drops activaron el bloqueo de historia; se produjeron
115/250 salidas, todas con paridad. Se preservaron 724 archivos. No admite
cobertura completa ni 50 Hz sostenidos y no prueba que la caché causase los picos.
**V2 aún no está integrada en fuentes/runner ni admitida en TTC causal.**
El [diseño de wiring pendiente](FROZEN_RGB_V2_WIRING_DESIGN.md) requiere otra
asignación al retomar; no se ejecutará durante este cierre.

El [micro CPU generado](FRAME_HASH_CPU_GENERATED.md) conserva el estado histórico
del experimento, anterior al core; sus cifras son de copia/hash aislados. El core
conserva identidad criptográfica del contenido y no usa IDs opacos. BLAKE3 redujo
el coste CPU medio de copia+hash de 3,9973 a 1,6342 ms/frame en ese micro, pero
**no existe una mejora end-to-end V2 medida**. No demuestra 50 Hz/20 ms ni admite
todavía el paso a dos objetos. El contrato aprobado se
conserva a continuación; las frases de integración futura siguen pendientes.

## Contrato de compatibilidad

`freeze_rgb_frame` seguirá siendo la única fábrica de `FrozenRGBFrame`. Sin opción
nueva seguirá usando SHA-256, el descriptor V1 y la misma serialización exacta:

```python
(source.name, source.revision_sha256, frame_id, timestamp_us,
 shape, "uint8_HWC_RGB", content_sha256)
```

No se añadirá un prefijo a ese descriptor V1. Sus bytes JSON, digest de payload,
`identity_sha256`, namespace de historia y comportamiento default permanecerán
idénticos. La revisión de fuente seguirá siendo una SHA-256 independiente; no
cambia por elegir otro algoritmo para los píxeles.

La misma fábrica aceptará un backend BLAKE3 previamente verificado, opcional y
explícito. No admitirá callbacks arbitrarios ni nombres que puedan declarar un
algoritmo diferente del ejecutado. La selección V2 conserva validación de dtype,
shape, bounds e IDs, copia C-order a `bytes`, registro privado autenticado con
referencias débiles, vistas readonly, owner/lifetime y sellado al llegar el frame.

Las propiedades nuevas serán `token_version`, `content_algorithm` y
`content_digest`. V1 informa `1`, `sha256` y la SHA actual; V2 informa `2`,
`blake3-256` y el digest BLAKE3 de 32 bytes. `hashed_bytes` cuenta los mismos bytes
RAW en ambos casos. `identity_sha256` sigue siendo realmente SHA-256 del descriptor
de metadata. `sha256` y `content_sha256` mantendrán su retorno V1 y lanzarán un
error explícito para V2. Consultarlas no calculará una SHA adicional oculta.

El descriptor V2 será explícitamente distinto:

```python
("IMMUTABLE_RGB_FRAME_V2", "blake3-256", source.name,
 source.revision_sha256, frame_id, timestamp_us, shape,
 "uint8_HWC_RGB", content_digest)
```

La misma serialización JSON determinista y SHA-256 exterior enlazarán ese
descriptor. Mismos píxeles y metadata producen la misma identidad dentro de V2;
cambiar contenido o cualquiera de esos campos la invalida. La identidad conserva
significado por contenido, no depende de la dirección del objeto ni del proceso.

## Archivos implementados y alcance

| Archivo | Cambio mínimo |
|---|---|
| `operational/train40_runtime/rgb_hash_backend.py` (nuevo) | Cargar únicamente el BLAKE3 privado fijado y verificado, exponer backend autenticado de un hilo y recibo de dependencia. Importación opcional, sin coste/dependencia BLAKE3 en V1. |
| `operational/train40_runtime/rgb_raw_ring.py` | Extender record/props y opción de fábrica; dejar rama V1 exacta; descriptor V2; ring rechaza perfiles mixtos y detecta transición de perfil. |
| `operational/train40_runtime/rgb_frozen_identity.py` | Mantener namespace V1; V2 usa `("IMMUTABLE_OWNED_RGB_V2", "blake3-256", source.name, source.revision_sha256)`; rechazar mezcla de perfiles en una historia. |
| `tests/unit/test_train40_runtime_rgb_hash_v2.py` (nuevo) | Fixtures de compatibilidad, identidad, ownership, separación de caches y contrato de dependencia. |

El backend se verificará una vez en setup, antes de medir llegadas y sin tocar
píxeles. API disponible: `load_verified_blake3(private_dependency_path)` devuelve
un backend factory-only; `freeze_rgb_frame(..., hash_backend=backend)` opta por
V2. `hash_backend=None` conserva V1. El backend solo representa la implementación
de hashing verificada; no sustituye el digest del contenido por una capacidad.
La llamada de hashing fijará siempre `max_threads=1`, `hexdigest(length=32)`.

El verificador no importará código de `scripts`. Comprobará un pin independiente
de versión, nombre y SHA del wheel, metadata/licencia, bytes instalados contra el
wheel, versión instalada y origen de módulo/extensión. Debe rechazar un módulo
ya cargado desde otro sitio y restaurar el path de importación tras la carga.
No habrá fallback a SHA ni descarga automática. Dependencia ausente o alterada
produce error accionable; no modifica venv, pyproject ni lockfile. El recibo
incluirá los paths que deberán entrar en los snapshots de un futuro ensayo.

Pin de este experimento: `blake3==1.0.11`, wheel
`blake3-1.0.11-cp311-cp311-win_amd64.whl`, SHA-256
`de3fbfeef38f68b32c23ae954a83bbfc0c69189c480b045f91ae55e0f0ef9007`.
No se presupone que ese wheel sea portable a otras plataformas. Procedencia,
licencia y fuentes primarias están en el
[recibo de dependencia](../../artifacts/train40_runtime_council_20261028/hash_probe_dependencies/README.md).

## Ring y caches

Una llamada de `bind` aceptará un solo perfil `(token_version, content_algorithm)`.
Rechazará mezclas antes de subir datos. Entre llamadas, un cambio V1/V2 reseteará
la ocupación y quedará registrado como `hash_profile_reset`, separado del reset
por cambio de fuente. Los receipts explicitarán el perfil y que las identidades
externas siguen siendo SHA-256 de metadata. El contador de bytes RAW hasheados
dentro de `bind` seguirá en cero.

La cache de crops ya usa `token.identity_sha256`; la cache de features usa el
digest/namespaces de historia. Esas claves separan V1/V2 automáticamente con los
descriptores propuestos. No necesitan modificar sus formatos ni sus políticas.
Se verificará esa separación mediante tests de hits/misses y contenido exacto.
`FrozenRGBHistory` puede seguir suministrando exclusivamente las vistas de tokens
autenticados, con el binder actualizado y sin escanear de nuevo los píxeles.

Los únicos consumidores actuales de `.sha256`/`.content_sha256` encontrados son
diagnósticos y tests; los caminos de producción usan payload/array/identidad.
Los diagnósticos históricos se mantienen V1 y no deben cambiar sus recibos.

## Contratos cubiertos por QA CPU y siguiente gate

- Fijar digest de contenido, descriptor e identidad V1 con fixture conocida;
  comprobar opción omitida y `None` con los mismos valores exactos.
- V2 determinista, digest cotejado con vector/API BLAKE3 y namespace diferente
  de V1; cambios de píxel, fuente, shape, timestamp o ID invalidan identidad.
- `content_sha256` y `sha256` fallan en V2; spy de SHA confirma que solo se
  hashea metadata pequeña y no el payload V2. Binder/ring no rehashean payload.
- Bytes propios inmutables y registry auténtico; mutar el original no cambia
  token; tokens falsificados y perfiles mixtos fallan antes de consumo.
- Ring CPU produce miss/reset al cambiar perfil y hit al repetir V2; la cache
  de crops y la de features no reutilizan entradas V1 bajo claves V2.
- Dependencia ausente, wheel/instalación alterados, origen erróneo y backend
  falso fallan; el default V1 funciona sin importar BLAKE3.

No se tocarán `paced.py`, `benchmark_train40_paced.py`, `multiobject.py`, controles
actuales ni selección de fuente en esta primera implementación. La integración
futura deberá propagar la opción coherentemente tanto a la fuente persistente
como a `replay_memory`; una bandera en un runner no basta si otra fuente sigue
sellando con SHA. Ese paso necesita coordinación de propietarios y nuevos
recibos explícitos de algoritmo.

Después se medirá la fábrica completa y un ensayo causal de un objeto, incluyendo
el sellado de todo RAW nuevo dentro del reloj, arranque separado, burst real,
preprocesamiento, transferencias, modelos, cobertura/paridad TTC y percentiles.
No se prehashearán observaciones futuras ni se descontará el micro de otro p95.
