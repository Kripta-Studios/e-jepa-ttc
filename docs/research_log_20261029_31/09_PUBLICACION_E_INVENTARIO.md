# 09. Publicación en GitHub e inventario de entrega

## Alcance autorizado

La petición actual solicita Markdown detallado y `git add`, `git commit` y
`git push` de todo el trabajo a GitHub. Se publica la rama existente
`research/evttc-transfer-v2`, incluyendo los commits de implementación previos,
la memoria nueva y los documentos de entrada todavía no versionados. No se
reescribe historia ni se fuerza una actualización de main.

Los ZIP propios se adjuntan a la prerelease
[`phase-ttc-research-20261031`](https://github.com/Kripta-Studios/phase-ttc/releases/tag/phase-ttc-research-20261031).
Esto mantiene pesos y paquetes completos accesibles sin convertir Git en almacén
de blobs grandes. El estado de prerelease expresa que se trata de investigación
con resultados locales y pendientes de evaluación externa, no de una versión
certificada para uso de seguridad.

El commit de código/modelo de la última entrega es `e482c220`. La publicación
documental tiene un commit posterior. Los hashes de los ZIP no cambian al
publicarlos, y no se reempaquetan para fingir que se generaron con el commit del
dossier. El tag fija el commit de la publicación documental inicial; posteriores
recibos de subida pueden añadirse a la rama sin mover ese tag.

## Qué entra en Git

- Código y tests de las iteraciones anteriores, mediante el historial de la rama.
- El dossier `docs/research_log_20261029_31/`, sus fuentes de evidencia pequeñas,
  inventario de assets y checksums.
- Los documentos de entrega en `docs/evttc_submission_20261031/` y las auditorías
  anteriores de `docs/evttc_transfer_v2/`.
- Los Markdown recibidos en la raíz y la carpeta extraída
  `PHASE_TTC_Ruta_SOTA_2026-10-30/`, conservados como documentos históricos.
- El enlace desde README y las reglas de exclusión de archivos de ejecución.

Los documentos de entrada mantienen sus afirmaciones y estado de aquel momento.
La cronología del dossier explica cuáles se ejecutaron, cuáles se revisaron y
cuáles siguen siendo propuestas. Añadir un plan a Git no convierte su contenido
en experimentos completados.

## Qué se conserva fuera de Git

`output/` contiene cachés, entornos de prueba de instalación, replays, copias
extraídas, investigación auxiliar y artefactos locales voluminosos. Los datos
crudos HDF5/bag no se redistribuyen. `.superpowers/` es estado temporal del agente.
Los archivos `arXiv-*.tar.gz` son fuentes de terceros consultadas, no código del
proyecto que deba republicarse. Los ZIP de raíz quedan fuera de Git porque se
publican como assets con nombre y digest.

La exclusión no oculta resultados negativos: los informes y paquetes conservan
los ensayos fallidos/rechazados y sus recibos. Los manifests describen la evidencia
de las carpetas de ejecución sin obligar a publicar sensores o entornos enteros.
La disponibilidad del raw dataset y sus condiciones de uso siguen correspondiendo
a sus proveedores.

## Inventario de assets

| Archivo | Bytes | Miembros ZIP | SHA-256 (prefijo) |
|---|---:|---:|---|
| `PHASE_TTC_ADAPT_F1_S7_2026-10-30.zip` | 27483815 | 24 | `da7014d0aac586d4` |
| `PHASE_TTC_ADAPT_FINAL_2026-10-31.zip` | 36023481 | 2589 | `4805fc06131988a9` |
| `PHASE_TTC_Adapt_EvTTC_results_2026-10-31.zip` | 13504 | 10 | `e7e41820003f18a2` |
| `PHASE_TTC_CAMPAIGN_FINAL_2026-10-30.zip` | 34963401 | 298 | `3f565fabe930aa5e` |
| `PHASE_TTC_EvTTC_results_2026-10-28.zip` | 13538 | 10 | `2c79847516993924` |
| `PHASE_TTC_Plan_SOTA_2026-10-30.zip` | 85267 | 21 | `4247a62948500cde` |
| `PHASE_TTC_ROBUST_FINAL_2026-10-31.zip` | 35864173 | 2598 | `0cc72d25174a99af` |
| `PHASE_TTC_SOTA_PLAN_AND_AUDIT_20261029.zip` | 83978 | 13 | `8237f47f3bdd61e5` |
| `PHASE_TTC_STARTUP_FINAL_2026-10-31.zip` | 65421612 | 2793 | `c499dc7549fba528` |
| `phase_ttc_g0_g1_review_bundle.zip` | 288053 | 46 | `75ec2ebd7fabb814` |

El inventario completo con hashes de 64 caracteres está en
[ASSETS.json](evidence/ASSETS.json); [SHA256SUMS.txt](SHA256SUMS.txt) permite
comprobar las descargas. La columna de miembros cuenta todas las entradas del ZIP,
incluidos README y manifiestos. No coincide necesariamente con el contador de
payloads rehasheados que aparece en un recibo anterior.

Los cuatro primeros tipos de material histórico son planes, revisión de G0/G1
y la submission original. Los paquetes F1, campaña, robust y startup contienen
modelos/evidencia de iteraciones diferentes. Para enviar el nuevo candidato al
benchmark, el archivo pertinente es `PHASE_TTC_Adapt_EvTTC_results_2026-10-31.zip`;
su acompañamiento técnico es `PHASE_TTC_ADAPT_FINAL_2026-10-31.zip`.

## Comprobaciones de publicación

Antes del commit se revisan enlaces del dossier, correspondencia de las tablas
generadas con JSON, integridad CRC de los diez ZIP y ausencia de sensores crudos
o archivos de claves en su inventario. El diff debe estar libre de errores de
whitespace. Una revisión independiente comprueba las afirmaciones y el alcance.

Después del push se compara el SHA de la rama remota con el commit local. Después
de subir assets se contrastan nombre, tamaño y digest de GitHub con el inventario
local; si la API no ofrece digest, se descarga el asset para rehashearlo. El
recibo [PUBLICATION.json](evidence/PUBLICATION.json) registra los resultados reales.
No basta con afirmar «push lanzado» o «upload iniciado».

Publicar una prerelease no envía correos ni presenta resultados al scorer de
EvTTC. Los borradores siguen disponibles para esa acción posterior. Tampoco
modifica la ficha original del benchmark ni convierte las métricas locales del
modelo adaptado en resultados oficiales.

Se conservan veinte saltos de línea Markdown de dos espacios al final en tres
documentos de entrada. `git diff --check` los señala como whitespace, pero
eliminarlos alteraría los bytes de las plantillas cuyo manifiesto se preserva.
El contenido nuevo pasa esa comprobación; la excepción queda limitada a esas
líneas históricas, que no se tratan como errores del código.
