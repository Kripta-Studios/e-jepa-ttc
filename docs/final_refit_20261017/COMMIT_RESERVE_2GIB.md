# Reserva de memoria comprometida: 2 GiB

Este documento describe V5. La continuación V6 reduce también la reserva
física a 2 GiB: [estado y cambio posterior](RAM_RESERVE_2GIB.md).

El 18-10-2026 el usuario autorizó reducir a 2 GiB el margen mínimo de memoria
comprometida disponible de Windows y reanudar ambos productores en paralelo.
El mínimo efectivo es ahora 2.147.483.648 bytes en el guard inicial, el cliente
del inventario compartido y el control final de los productores. La reserva
de RAM física del entrenamiento sigue siendo 6 GiB; se mantienen las
comprobaciones de disco, cobertura del inventario y checkpoints cada 100 updates.

La pausa anterior ocurrió con 7,46–7,68 GiB de margen de commit, por debajo del
antiguo mínimo de 8 GiB. Ambos guardaron íntegramente su estado:

| Productor | Update durable | SHA-256 del checkpoint conservado |
|---|---:|---|
| R_A5 | 40.673 | `d55401f781de3c95f7e7f5e9b70242823dc7b19111326fb6086be939624649ee` |
| R_C2F | 22.475 | `af11e29dc75832c6191230e14a73579c1d870733ca2e5d46ac3d08984ad7495e` |

El runtime V5 conserva los bytes de esos checkpoints, las curvas y la identidad
científica. Sólo cambian tres constantes de los controles de commit. No cambian
la aritmética del entrenamiento, RNG, optimizer, scheduler, datos ni endpoint.
La comparación de las fuentes contra el baseline verifica exactamente esas
tres sustituciones; no se presenta como una nueva matriz numérica de GPU.

Verificación focalizada: 48 pruebas aprobadas, Ruff limpio y Pyright sin errores
usando el entorno real del proyecto. Los tests verifican aceptación en 2 GiB y
rechazo por debajo, además de conservar el resto de controles. La migración de
ambos checkpoints pasa con recovery_upper=0 y cero optimizer updates adicionales.

Evidencias: `artifacts/final_refit_20261018_commit2gib/CHANGE_PROOF.json`,
`UNIT_VERIFICATION.xml` y `MIGRATION_RESULT.json`.
El runtime vigente está en `artifacts/final_refit_20261018_runtime_v5` y el
localizador operativo es `artifacts/final_refit_20261017/ACTIVE_RUNTIME.json`.
El lanzador es `RESUME_CONCURRENT_RGB_V5.ps1` en la carpeta de evidencias.
No se han cambiado ajustes globales de Windows ni su archivo de paginación.
