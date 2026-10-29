# Reserva física y de commit: 2 GiB

El 18-10-2026 el usuario autorizó reducir también el mínimo de RAM física
disponible de 6 a 2 GiB y reanudar el entrenamiento. La reserva de memoria
comprometida disponible de Windows permanece en 2 GiB. Ambos umbrales equivalen
a 2.147.483.648 bytes. El guard inicial y el monitor compartido ya comprobaban
2 GiB de RAM; se actualizó el control adicional de los productores.

A5 había guardado el update 40.728 al detectar 5,98 GiB de RAM física disponible.
C2F seguía activo y se guardó expresamente en 22.858 para relanzar ambos bajo el
mismo runtime. Los dos estados tienen recovery_upper=0 y ningún update pendiente.

El runtime V6 conserva los checkpoints, las curvas y la genealogía científica.
La única diferencia en sus fuentes es el umbral físico `6 → 2` del guard final.
Los objetivos, datos, orden de muestras, RNG, optimizer, scheduler y endpoint
permanecen idénticos. No se ejecutó una nueva matriz numérica de GPU para este
cambio de control: la prueba compara las fuentes y verifica esa única sustitución.

Validación: 48 tests focalizados aprobados, Ruff limpio y Pyright sin errores.
El test del guard final acepta exactamente 2 GiB y rechaza un byte menos.
Se mantienen los controles de disco e inventario y checkpoints cada 100 updates.

Evidencias: `artifacts/final_refit_20261018_ram2gib/CHANGE_PROOF.json`,
`UNIT_VERIFICATION.xml`, `MIGRATION_RESULT.json` y el lanzador
`RESUME_CONCURRENT_RGB_V6.ps1`. El runtime es
`artifacts/final_refit_20261018_runtime_v6`. El estado vigente está en
`artifacts/final_refit_20261017/ACTIVE_RUNTIME.json`.
