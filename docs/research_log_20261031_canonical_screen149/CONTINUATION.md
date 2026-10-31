# Continuación activa tras el screen

Registro a 2026-10-31T18:26:55.968736+00:00. No es una pausa ni un cierre del
objetivo: continúa la investigación hacia evidencia reproducible bajo protocolo
comparable, preservando eAP. SOTA sigue sin acreditarse.

## Ejecución en marcha

- Código de extracción completo publicado: `4d9b5b7a`.
- Run: `output/phase_timewarp/population2608_v1`.
- Log: `output/phase_timewarp/population2608_v1.stdout.log`.
- Sello inicial SHA-256: `a847ace2866628af3b0e9e0ccb91038fc4da93593ffd56d0f6f5d746229623f1`.
- Sesión de ejecución de esta conversación: `59503`.
- PID observado: 77708 (worker); 70352 (launcher). Verificar que siguen siendo
  estos comandos antes de actuar; los PID pueden reutilizarse.
- Última línea observada al escribir este registro:

```text
192/2608 new=187 evttc:EXTENDED:CCRm-low-50-overlap-50:14400369 3.14s elapsed=457.5s rolling_ETA=103.9min
```

Se reutilizan 149 consultas y se extraen 2459 nuevas de carretera, sin targets
por consulta. F0 es la ruta primaria que pasó el screen. El router queda como
diagnóstico tras su regresión; no se promueve mediante el éxito de F0.

No iniciar otra copia mientras exista este proceso. La existencia de FREEZE o
de consultas individuales no significa que haya terminado. Esperar COMPLETE.json
y verificar sus hashes; FAILED.json exige conservar el intento y diagnosticar.
El runner soporta --resume únicamente con fuentes/entradas idénticas y consultas
completas selladas. Rechaza carpetas parciales o runs declarados fallidos.
No modificar fuentes incluidas en FREEZE mientras ejecuta: la verificación final
rechazará cambios. Se guarda una copia exacta de las fuentes ejecutadas.

## Evaluación posterior al sello

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.venv-transfer/Scripts/python.exe scripts/evaluate_phase_canonical_screen.py --screen output/phase_timewarp/population2608_v1 --expected-queries 2608 --output output/phase_timewarp/population2608_evaluation_v1
```

El evaluador rechaza inferencia incompleta o alterada antes de abrir targets.
Incluye macro por grabación, cobertura, signos, los mismos criterios fijos y
bootstrap pareado de grabaciones en carretera (semilla 7, 10000 remuestreos).
El generador report_phase_canonical_screen.py es específico de las 149 consultas;
no usar su texto fijo para describir la población completa. Crear el informe de
2608 desde el nuevo REPORT.json cuando exista.

## eAP y trabajo concurrente

La inspección de recibos del archivo final_generated confirmó OOF_AUTHORITY en
E_CTX_MATCHED y R_CTX. El inventario reproducible está en
`output/phase_timewarp/eap_history_inventory_v1/INVENTORY.json`; copia pequeña en
`evidence/eap_history_inventory.json`. Regeneración:

```powershell
.venv-transfer/Scripts/python.exe scripts/inventory_phase_eap_history.py --archive-root D:/E_JEPA_POD2_RETURN_20261027 --input-freeze output/phase_timewarp/eap_current_pair_audit/INPUT_FREEZE.json --output output/phase_timewarp/eap_history_inventory_new/INVENTORY.json
```

No se han recalculado historias eAP completas. No mezclar estas cachés OOF con
PAIR133 de productores de despliegue para afirmar conservación. La diferencia
de cadencia RGB y las ROIs condicionadas por consulta requieren genealogía
consistente. Test12 continúa sin targets.

Apareció trabajo concurrente en docs/physics_review_20261031. No fue creado por
esta ejecución y se preservó sin añadirlo al commit. Se consultaron README y
10_PLAN_EXPERIMENTAL_ACOTADO: coinciden en terminar primero el screen/población
y proponen después observabilidad y mecanismos, sin reabrir JEPA genérico.
No interpretar la presencia de esa carpeta como fallo de limpieza de este run.

## Validación adicional

La captura compacta y el replay pasaron 7 pruebas focalizadas; el evaluador
ampliado pasó 6 pruebas, incluyendo bloqueo previo a targets y bootstrap por
grabación. Ruff y Pyright pasan en los archivos nuevos/cambiados. Las 49 pruebas
del screen y su salvedad de formato heredado están en VALIDATION.md.

El entregable sigue intacto. No hay nueva entrega oficial ni correo enviado.

## Avance posterior mientras continúa la extracción

Se verificó vivo el worker 77708 al alcanzar 338/2608, 330 consultas nuevas,
830.2 s transcurridos y ETA móvil de 75.0 minutos. El ETA varía con la escena;
no es una hora de finalización garantizada. La sesión sigue siendo 59503.
Los 55 hashes de fuentes del FREEZE continúan intactos.

Ya está preparado `scripts/report_phase_canonical_population.py`, específico
para 2608: valida sellos y ámbito antes de publicar, conserva F0 primario y
router diagnóstico, e incluye denominadores, grabaciones y bootstrap. Tras
ejecutar el evaluador indicado arriba, usar:

```powershell
.venv-transfer/Scripts/python.exe scripts/report_phase_canonical_population.py --screen output/phase_timewarp/population2608_v1 --evaluation output/phase_timewarp/population2608_evaluation_v1 --output docs/research_log_canonical_population2608
```

No se ha generado ese informe ni abierto targets de la población en curso.
El generador rechaza tanto el screen de149 como el run incompleto de2608;
recibo en `output/phase_timewarp/report_population_validation_v1/VERIFICATION.json`.
Ruff, formato y Pyright pasan en ambos scripts nuevos.

Nueva evidencia de eAP: [EAP_EVENT_CADENCE.md](EAP_EVENT_CADENCE.md). Los 88.744
deltas del índice de históricos son exactamente float32(0.1), idénticos a los
de las consultas actuales. La ruta inspeccionada repite ese delta en sus
históricos: se cumple la precondición de la rama de identidad de CanonicalPair.
Es evidencia estructural; no una validación ejecutada de contextos/cabeza ni
un motivo para omitir la comprobación de genealogía de despliegue y RGB.

## Disponibilidad posterior de la ruta Test12

El preflight [EAP_DEPLOYMENT_PREFLIGHT.md](EAP_DEPLOYMENT_PREFLIGHT.md) localizó
las 5.805 entradas físicas de eventos y los 36 shards RGB con tamaños esperados.
Se verificaron los hashes de los 12 archivos de diagnósticos y la coincidencia
de genealogía de su recibo con el bundle entregado. Los hashes de medios aún
deben verificarse antes de usarlos. No se ejecutó ningún candidato ni se
abrieron targets Test12.

Los 5.805 deltas de eventos son identidad; los 5.805 deltas RGB no lo son.
Los diagnósticos conservan contexto/cabeza y raw17, pero faltan PAIR133. Las rutas,
comando y evidencia están en ese documento. No mezclar con las cachés OOF.

La extracción de población seguía viva al alcanzar 443/2608 (434 nuevas),
1255.8 s transcurridos. El ETA móvil fue 151.5 minutos; había subido hasta
288 minutos con consultas de 10 s y luego volvió a bajar. No usar el ETA
anterior de 65–75 minutos como compromiso. La sesión continúa siendo 59503,
worker 77708, y sigue siendo necesario esperar el sello completo antes de
abrir targets. No reiniciar por un cambio de velocidad.

Apareció además `docs/release_research_plan_20261031/`, trabajo concurrente no
creado ni añadido por esta ejecución. Preservarlo igual que physics_review.

## Verificación de medios eAP en marcha

Código publicado en `381c9a2b`: `scripts/verify_phase_eap_media.py`.
Se está verificando SHA-256 de los 5.805 tensores de eventos y 36 shards RGB
contra el inventario público congelado, sin decodificar arrays ni ejecutar
modelos. Run: `output/phase_timewarp/eap_media_verified_v1`; log:
`output/phase_timewarp/eap_media_verified_v1.stdout.log`.

Sesión 3801; PID worker observado 71148, launcher 77168. Revalidar el comando
antes de actuar; no iniciar otra copia. FREEZE.json no es un resultado final.
El último avance observado es 1300/5841 archivos, 6.201 GB, 119.0 s; todavía
no se ha verificado la población completa. Esperar COMPLETE.json o conservar
FAILED.json si aparece. La fuente del verificador tampoco debe modificarse
mientras corre: compara su hash al cerrar. El manifiesto completo de paths y
hashes queda local; publicar solo el recibo compacto cuando termine.

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe -u scripts/verify_phase_eap_media.py --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --media-root E:/eAP_dataset --output DIRECTORIO_NUEVO
```

En la misma observación, EvTTC llegó a 516/2608 (505 nuevas), 1597.4 s,
ETA móvil 237.2 min. Sigue vivo el worker 77708, sesión 59503. Mantener el
protocolo y las fuentes intactos; evaluar solo tras el sello completo.
