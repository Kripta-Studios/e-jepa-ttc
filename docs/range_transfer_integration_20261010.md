# RANGE-TRANSFER: integración V14

Base de código: da068ebdeb2d967ee10b2afa348e1317b8720ad0.
Evidencia histórica: V12 1a686e338a1145c6794bec0704bdae4cda95f06d.
Rama separada: scientific-recovery-v14-range-transfer.

El usuario autorizó esta campaña el 10 de octubre de 2026. Los documentos
originales del handoff se conservan sin cambios en
[range_transfer_handoff_20261010](range_transfer_handoff_20261010/README.md).
El ZIP de entrada tiene SHA256
96a95c1ad8d63cbe5097e96b004c7f0d5f67e6a7707deb1ef6a7189e832ebb78.
También se leyó íntegramente readmenow.txt en V12.

## Ejecución y reentrada

Desde esta raíz V14, PowerShell 7:

```powershell
& pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1
```

El comando usa el Python existente del proyecto, mantiene la configuración del
equipo y escribe únicamente bajo V14. Devuelve 3 para dependencias/pausas,
con NEXT_DECISION.json y PENDING_DEPENDENCIES.json. No es un daemon ni
una tarea programada. Cada invocación observa nuevamente V13, verifica recursos,
ejecuta tareas viables y guarda una frontera durable.

```powershell
# Sólo diagnósticos; opcionalmente verifica/reutiliza la prueba técnica contabilizada.
& pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1 -DiagnosticsOnly -SelfTest
# Limita esta invocación sin acortar los endpoints registrados.
& pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1 -MaxUpdatesThisCall 100
```

El factorial se admite exclusivamente con finales E_A5_MATCHED y E_C2F_MATCHED
de 30330 updates y PAIR_E_MATCHED de 6840, ligados al rol P; caché H, normalizador
y admisión nativos íntegros. No usa checkpoints parciales ni productores TRAIN40.
La propiedad de V13/R1, su cola y sus admisiones permanecen intactas.

## Qué está implementado

- M0/M1/M2 y L0/L1, sampler uniforme común, inicialización común por seed,
  GRU160 nativa, FP32 CPU, batch128, scheduler y decodificador originales.
- Contabilidad 25000 científicos, 500 técnicos, 4500 recuperación y 30000
  físicos como máximos; ninguna reasignación de ahorro.
- Journal por update, checkpoint durable, bloqueo de escritura V14,
  pausa por recursos, continuación sin empezar fits de nuevo.
- Congelación de seis endpoints antes de abrir V; gates fijos y una sola receta;
  réplicas candidato/control seeds13/23 y bootstrap de grupos.
- Pérdidas por consulta promediadas entre seeds; no ensemble de TTC.
- Diagnósticos de masa/exposición H y atribución de cabezas V12 en consultas
  FCWD ya publicadas. Traza cruda V12 aislada, sin entrenamiento.
- Informe, figuras, manifest de hashes y ZIP esencial verificable.

## Evidencia ejecutada y límites

Los resultados fechados están en artifacts/range_transfer_20261010.
QA ejecutó 20 updates sintéticos: 10 continuos frente a 5+restore+5, con
igualdad exacta de parámetros, AdamW, scheduler, RNG, draws y pérdidas.
Los pesos QA no son pesos científicos.

H tiene 18422 filas y ocho grupos; sólo 52 filas con TTC hasta 1 segundo.
La masa nativa de 0,1–3 segundos es aproximadamente 10 %, frente a 50 % en M1.
Esto describe exposición/contribución, no demuestra una mejora.

Se ejecutaron 96 forwards CPU de cabezas históricas V12 en 32 consultas
FCWD seleccionadas, con diferencia máxima 4,77e-7 segundos frente al replay
guardado. Se conservan comparadores, exclusiones y resultados negativos.
FCWD es diagnóstico público de transferencia históricamente observado:
no confirmación externa ni SOTA universal, y nunca fuente de fit/selección.

El 11 de octubre se admitieron los tres padres finales y la caché H nativa:
18422 filas, dominio íntegro y normalizador conservado. Los gradientes H y los
32 controles de cabezas matched están completos. Los seis fits terminaron sus
2500 updates y quedaron congelados antes de evaluar las 16432 consultas V.
SELECTION_FREEZE.json registra NO_CANDIDATE: ninguna variante supera todos
los gates. Todas fallan la salvaguarda de rango largo. No se ejecutan réplicas
ni transferencia de endpoints nuevos; no se reasigna el presupuesto restante.
Contabilidad final: 15000 científicos, 20 técnicos, cero recuperación; 15020
físicos. Los resultados y el bootstrap por grupos se conservan íntegros.

Las trazas adicionales A5/C2F recorrieron 118 observaciones reales de los
32 controles H. Se conserva el fallo del adaptador PAIR, que exige una máscara
known con la forma del soporte temporal aunque el productor emite known por
consulta. El error también está comunicado en Dev32. Un adaptador diagnóstico
separado en qa/matched_pair_adapter.py valida la forma real por consulta y
conserva exactamente las 133 features nativas. Su prueba de contrato pasa y
la traza adaptada completa A5/C2F/PAIR en las 118 observaciones, sin optimización.
El delta de fase máximo CPU frente a la caché GPU es 6,704e-5.
No se cambia V13, el código científico congelado ni las cachés H/V.
La transferencia de nuevos endpoints necesita features matched admitidas,
y sólo procede tras selección; nunca usa features de productores TRAIN40.

## Verificación local

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m pytest tests/test_range_transfer.py tests/test_range_transfer_integration.py -q --override-ini addopts=
& '../e-jepa-ttc/.venv/Scripts/ruff.exe' check operational/range_transfer src/e_jepa_ttc/range_transfer scripts/range_transfer_v12_trace.py tests/test_range_transfer.py tests/test_range_transfer_integration.py
```

Los tests de contrato no ejecutan optimizer.step. La única prueba con updates
es QA, contabilizada y reentrante. No se ejecutó la suite completa del repositorio.
Pyright usa --pythonpath del entorno existente; la ausencia de .venv en V14
no autoriza instalar ni cambiar el entorno global.
