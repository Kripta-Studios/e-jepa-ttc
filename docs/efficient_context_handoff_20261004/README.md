# E-JEPA-TTC — siguiente campaña: contexto eficiente y comparación justa

**Fecha de revisión: 4 de octubre de 2026.** Base reproducible: tag
`simplex-t-local-results-20261004-complete`, commit
`fd16d8102914537653622d3abf298b753120ea43`.

## Decisión
Conservar H8 como candidato histórico y H16 como challenger. No repetir T6, H16
ni la campaña nocturna. La próxima campaña combina (1) acelerar la preparación
sin alterar predicciones, (2) una única prueba H8-WIDE para separar alcance temporal
y densidad, y (3) obtener comparadores Garl realmente ligados a TRAIN y al contexto.
No se promete tiempo real ni un resultado SOTA antes de medirlo.

## Lo que este paquete hace y lo que no
Contiene protocolos, prompts, preparación segura de worktree, verificación de
archivos y contratos de referencia con tests. **No es un trainer de eAP ya integrado.**
Codex debe integrar el runner con el repositorio, probarlo y ejecutarlo. Aquí no se
han entrenado modelos nuevos, accedido a E:, ejecutado CUDA ni validado PowerShell
sobre Windows. El intento local de clonado falló por errores de acceso de red. El código se inspeccionó
por GitHub conectado. Los seis ZIP de la release no se pudieron descargar aquí;
se verificó el ZIP histórico T6 ya adjunto y se analizaron sus índices. No confundir
recibos publicados con una reproducción independiente realizada en esta revisión.

Leer en este orden:
1. `docs/01_RESULTADOS_Y_DECISION.md` y `docs/02_IDEAS_Y_SEMANTICA.md`.
2. `docs/03_RUNTIME.md`, `docs/04_H8_WIDE.md`, `docs/05_GARL.md`.
3. `docs/06_EJECUCION_Y_ENTREGA.md` y `PROMPT_CODEX_START.md`.

## Autorización al enviar el prompt
E0/E1: implementación, auditoría acotada, control EWMA con transporte temporal
y perfilado, cero updates científicos.
E2: H8-WIDE seed 7 × tres folds, 7.500 updates; replicación condicionada seeds 13/23,
15.000 adicionales. E3: recuperación de Garl existente primero; si no hay productores
admisibles, hasta 12 productores Garl nuevos bajo un máximo conjunto de 200.000
updates y hasta seis cabezas Garl-H1/H8, 15.000 updates. Máximo científico conjunto:
**237.500 updates**. Máximo técnico sintético: 200; reserva física de recuperación:
2.300. **Techo físico global: 240.000**, nunca una bolsa para inventar más brazos.
Se calculan los pasos exactos Garl antes del primer update; si 50 épocas de todos los
fits no caben, esa rama se registra como bloqueada por presupuesto, no se acorta.

`execution_policy.json` es la autoridad de este paquete cuando el usuario lo envía
como autorización nueva. El freeze y presupuestos históricos NO se editan.
La rama Garl es independiente de la utilidad de H8-WIDE. Una dependencia que bloquee
Garl no bloquea optimización ni la prueba WIDE basada en cachés.

## Arranque recomendado
Extraer el ZIP en una carpeta nueva fuera de los worktrees históricos. En PowerShell7:

```powershell
& .\scripts\Prepare-Worktree.ps1 -SourceRepo 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v10-simplex-t-companion'
```

El script comprueba el paquete, crea un worktree separado a partir del commit
publicado y copia el handoff a una carpeta nueva. No hace reset, clean, pull ni push;
no descarga archivos salvo que se use `-AllowFetch` para recuperar un commit ausente.
Abrir Codex en el directorio que imprima y pegar `PROMPT_CODEX_START.md`.
En una sesión existente: enviar el prompt como nueva autorización, pasar de fase en
un límite seguro y no modificar ficheros de un fit activo.

## Comandos que ya existen en este paquete
```powershell
python .\scripts\verify_package.py --root .
python -m unittest discover -s tests -v
python .\scripts\verify_release_assets.py --directory 'RUTA_A_LOS_ZIP' --deep
```
`--download` en el último script permite descargar únicamente los seis archivos
inventariados. No es necesario si ya existen localmente. Los archivos correctos se
reutilizan; uno existente con hash incorrecto provoca error y no se sobreescribe.
No extrae ni ejecuta automáticamente código de un ZIP.

El comando del futuro runner `python -m operational.efficient_context.run ...`
**todavía no existe**: es una interfaz contractual que Codex debe implementar antes
de usarla. No ejecutar a ciegas launchers antiguos que reabran campañas cerradas.

Para regenerar únicamente mi análisis de índices históricos:
```powershell
python .\evidence\audit_windows.py --archive "RUTA_AL_ZIP_T6" --output "..\WINDOW_REPLAY.json"
```
El output debe guardarse fuera del paquete verificado, o en un directorio de trabajo
separado, para no alterar su manifiesto.

## Pruebas ejecutadas aquí
39 tests CPU de contratos y verificación de archivos aprobados. Se regeneró el
análisis de índices T6 con el script portable y coincidió exactamente con el
resultado guardado. Estos tests no son una validación de CUDA, del trainer
integrado o del script PowerShell en Windows. Ver `evidence/REVIEW_SCOPE.json`.
