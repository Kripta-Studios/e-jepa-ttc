# Alcance del archivo piloto congelado y del código posterior

`RGB_PILOT.tar` es una instantánea cerrada del perfil `rgb_pilot`:

- 10.569.820.160 bytes, 7.093 archivos.
- SHA-256 `a3dadc7e37e35735c9dad2d70bae5d3e28634a2dd6b802fe10d57ae3f2821cf0`.
- Código, referencias CPU y manifests incluidos se identifican dentro de `BUNDLE_MANIFEST.json`.
- Sirve para bootstrap, preflight y prueba de productor RGB tras autorización concreta.
- No contiene crudos para `finish-rotation`; ese comando rechaza el perfil piloto.

Después de congelar ese tar se completó en el código local la clausura de las cabezas:
`OOF_RECIPE.source_sha256` también debe viajar en el perfil `rotations`. Sus archivos históricos
se incluyen como evidencia, aunque no sean entradas numéricas de la rotación.

El empaquetado omite `.git`, credenciales y toda la historia del repositorio. Para las cabezas,
el adaptador registra el commit base original y lo devuelve como procedencia al logger congelado.
El propio binding deja explícito que el commit es solo la base: los hashes de fuentes y del bundle
identifican las modificaciones/untracked que realmente se ejecutan. No se afirma un checkout limpio.

Estos cambios no alteran el tar piloto publicado. El inventario posterior se conserva por separado
en `INVENTORY_FULL_V2.json`. Si se decide utilizar el código nuevo para otro piloto, generar un
archivo con nombre nuevo y volver a validar su binding; no cambiarlo debajo de un run ya iniciado.

Ninguno de los dos perfiles equivale a haber probado Linux/CUDA ni a tener la integración final
con el coordinador Windows. El paquete piloto permite comenzar esa validación sin trasladar
cientos de GiB ni contratar varias GPUs simultáneamente.
