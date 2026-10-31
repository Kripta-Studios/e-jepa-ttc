# Conservación antes de eliminar el pod

**JEPA: respaldo completo verificado.** Véase el
[recibo de verificación](retention_verification.json). El pod completo aún
no se declara listo para borrado porque la campaña DFULL seguía activa.

El archivo de cierre inicial contenía los resultados finales, pero no todas las
características calculadas ni los estados intermedios. Este respaldo adicional
conserva el contenido necesario para reutilizar el trabajo sin repetir esos
cálculos. El estado definitivo de la comprobación está en
`artifacts/jepa_council_20261031/preservation/LOCAL_VERIFIED_COMPLETE.json`;
no debe considerarse terminado sin ese recibo.

## Alcance y deduplicación

Se inventariaron por SHA-256 12.868 archivos del árbol
`/workspace/jepa_research_20261031`, que representan 58.688.410.975 bytes en sus
rutas originales. Para 12.195 rutas ya existía contenido idéntico en archivos
locales o miembros de archivos TAR/ZIP existentes. Las otras 673 rutas se
resuelven mediante 664 contenidos únicos: nueve rutas comparten contenido.

Solo se transfieren los 664 contenidos nuevos, 3.903.906.599 bytes útiles.
Se guardan una sola vez en `preservation/unique_payload.tar` (3.905.105.920 bytes
con cabeceras TAR), indexados por hash. El mapa `RESTORE_MAP.json` relaciona
cada ruta original con el archivo local o miembro de archivo que la conserva.
No es un respaldo autónomo que permita borrar después los archivos reutilizados.

Se conservaron:

- Los nueve checkpoints finales y todos los checkpoints de recuperación
  disponibles, con estados del optimizador y registros de entrenamiento.
- Características P/H de cada brazo y semilla, inicializaciones aleatorias,
  salidas de productores y características V-JEPA espaciales y globales.
- Datos de eventos preparados, recortes RGB, targets del profesor, planes,
  alineamientos temporales y pruebas del piloto de 32 y de 768 muestras.
- Los pesos completos y compactos del profesor, su código y licencias;
  los productores utilizados, código ejecutado, protocolos, logs y resultados.
- El inventario de 68 paquetes del entorno efectivo, Python y plataforma.
  El entorno registró Python 3.11.18, PyTorch 2.11.0+cu128, NumPy 2.4.6,
  pandas 3.1.0 y timm 1.0.30. GPU observada: A40, driver 595.91.07.

Se excluyen instalaciones reconstruibles de paquetes (`.venv`, `teacher_setup/deps`),
cachés `__pycache__`/pytest y bases de datos `.git`. Las fuentes de los repositorios
y las versiones quedan conservadas. No se copian otra vez los datos originales
ya disponibles localmente. El inventario enumera explícitamente las exclusiones.

## Ubicaciones que deben conservarse

El directorio principal local es:

```text
artifacts/jepa_council_20261031/preservation/
  unique_payload.tar
  PAYLOAD_RECEIPT.json
  RESTORE_MAP.json
  remote_inventory.json
  local_inventory.json
  EXISTING_VERIFIED.json
  NEW_VERIFIED.json
  LOCAL_VERIFIED_COMPLETE.json
```

También deben mantenerse el archivo de cierre anterior y los archivos existentes
referenciados por el mapa. Parte de los datos reutilizados está en unidades
locales distintas del repositorio, especialmente E:. El mapa registra sus rutas
absolutas. Si se trasladan a otro equipo, hay que trasladar esas dependencias y
actualizar el mapa; no basta con copiar únicamente el nuevo TAR.

No se han añadido datos crudos ni checkpoints a Git. Los archivos físicos de
respaldo están en el workspace y en las unidades locales indicadas por el mapa.
El informe y la herramienta de restauración sí pertenecen a la documentación
y al código del repositorio.

## Verificación y restauración selectiva

La herramienta usa únicamente la biblioteca estándar y nunca entrena:

```bash
python -S scripts/restore_jepa_pod_backup.py --manifest artifacts/jepa_council_20261031/preservation/RESTORE_MAP.json --verify
```

La verificación lee los bytes de cada archivo o miembro reutilizado una sola vez
y comprueba SHA-256. `--storage existing` y `--storage new` permiten verificar
por separado ambos conjuntos, cuya unión cubre todas las rutas inventariadas.
`--prefix` permite limitarse a una ruta original:

```bash
python -S scripts/restore_jepa_pod_backup.py --manifest artifacts/jepa_council_20261031/preservation/RESTORE_MAP.json --prefix causal_pilot768/seed7/ --destination RESTAURADO
```

La restauración verifica el hash de cada salida y rechaza sobrescribir un
archivo existente diferente. Sin `--destination`, no materializa copias.
Se probó una restauración real de un archivo pequeño desde el archivo existente;
el archivo de prueba se retiró después de comprobarlo.

## El borrado del pod completo depende de otro trabajo

Este respaldo cubre JEPA. El pod también contiene una campaña DFULL ajena a este
piloto, observada en estado RUNNING durante la conservación. Su respaldo automático
ya está activo en `D:\PHASE_TTC_DFULL_20261031`; no se duplicó ni se detuvo.
Las campañas anteriores y rare recovery tienen sus propios recibos locales en
`D:\PHASE_TTC_POD_BACKUP_20261031` y `D:\PHASE_TTC_RARE_RECOVERY_20261031`.

**Un respaldo completo de JEPA no autoriza a afirmar que se puede eliminar el
pod mientras DFULL sigue entrenando.** Antes de borrarlo hay que cerrar esa otra
campaña y comprobar su último respaldo verificado. Un snapshot RUNNING no
conserva actualizaciones posteriores ni el estado aún no guardado en memoria.
No se borró el pod ni se interrumpieron sus procesos durante este trabajo.
