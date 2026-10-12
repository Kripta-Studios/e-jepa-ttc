# V15 RANGE-READOUT: integración

Campaña nueva autorizada el 12 de octubre de 2026, sobre
`1c251781ac845a5d661523e607e0f7c4a8d3af8a`. V14 permanece cerrado
`NO_CANDIDATE`; V13 y R1 son dependencias de sólo lectura. Este documento describe
la implementación; el estado y los resultados proceden de los artefactos ejecutados.

## Cola prospectiva

| Familia | Fits seed7 | Updates por fit | Intervención |
|---|---|---:|---|
| S | S-M0-L0, S-M1-L0 | 2.500 | Muestreo con reemplazo p proporcional a la masa; estimador w/p |
| R | R-C05, R-C20 | 6.000 | Scorer residual A5; límite .05/.20 con pendiente inicial .05 |
| R-CTX | R-C05-CTX, R-C20-CTX | 2.500 | PHASE17 y normalizador H reconstruidos, cabeza nativa M0/L0 |

Máximos: 56.000 científicos, 500 técnicos sintéticos, 3.500 recuperación,
60.000 físicos. Sólo S-M1 y R-C20-CTX pueden promover su familia; deben pasar
los gates V14 tanto frente al control pareado como frente al M0/L0 histórico.
Las cuatro cabezas y sus padres se congelan antes de V. La réplica, si procede,
usa seeds13/23 para una sola familia. No se entrenan encoders.

## Contratos implementados

El scorer usa `equal_slope_tanh_v1`, con límite y pendiente explícitos en el
checkpoint. El loader comprueba el operador solicitado. Los dos brazos copian
el mismo scorer del padre y reinician a cero su Linear final. Encoder, transporte,
geometría, incertidumbre, proyector PAIR y auxiliares permanecen congelados.

La caché guarda vectores anteriores al residual. La ruta vectorial conserva la
aritmética posterior del productor: recomputa log-ratio, transporte temporal,
TTC, known, propagación de varianza y riesgo. Usa la pérdida nativa sobre el batch
efectivo completo, incluido top-k global. Las salidas geométricas y relaciones DINO
son constantes; sus contribuciones no desaparecen del valor de la pérdida.

Se verifican outputs, pérdidas y gradientes contra raw en P, además de T2/T3 y
negativos sintéticos. El contrato de inferencia rechaza campos de targets. PAIR
conserva sus 133 entradas al cambiar sólo el scorer. Sus predicciones históricas
se reutilizan; fase A5, margen, incertidumbre y desacuerdos PHASE17 se reconstruyen.
Cada normalizador se ajusta a observaciones H únicas.

La extracción CPU inicial se conserva como piloto. Después de observar recursos
y propietarios, se verificó extracción GPU FP32 contra readout CPU FP32 con
atol/rtol 2e-5. TF32 está desactivado en ese proceso. La optimización de lectura
reutiliza GroupRowCache nativo con paridad exacta de todos los campos de batch;
no cambia imágenes, filas ni orden. Los fits siguen en CPU FP32 y un thread numérico.

## Ejecución y continuidad

Desde la raíz del worktree V15:

```powershell
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1 -Help
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1
```

Hay un único coordinador V15. La maquinaria de checkpoint, journal y lock se
reutiliza de V14, con archivos alojados exclusivamente en la nueva campaña.
La extracción conserva fragmentos con hash y vuelve a comprobarlos al reanudar.
Un fallo se registra y detiene el stage; no se relanza indefinidamente.

Los tests de continuidad comparan pesos, optimizador, scheduler, draws y RNG.
Las pruebas técnicas no pueden recibir una caché científica. Su consumo figura
en el mismo ledger y no se oculta dentro de pytest.

## Evidencia

Raíz relativa: `artifacts/range_readout_20261012/`.

- `fits/`: pesos, curvas, journals y procedencia.
- `vector_cache/`: entradas congeladas por rol y fragmento.
- `families/`: PHASE17 y normalizadores coherentes con cada readout.
- `qa/`: contratos, paridades, continuidad y fallos conservados.
- `evaluation/`, `bootstrap/`: predicciones y contrastes tras el freeze.
- `ACCOUNTING.json`, `NEXT_DECISION.json`, `CAMPAIGN_STATE.json`: estado real.

V es desarrollo interno ya expuesto. FCWD/Dev32 sólo pueden ejecutarse después
de una réplica favorable; no entrenan ni seleccionan la receta. No se hacen
pushes, merges, submissions ni cambios globales del equipo en esta campaña.
