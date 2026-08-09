# Progreso

## 2026-08-09 — Causal Scale v5

- preservado el resultado negativo v4.31 y descartado añadir otro readout al matcher;
- implementado un core común de foreground/altura/log-ratio para eventos y futuro RGB;
- eliminados bbox, categoría e ID de secuencia de la API del candidato v5;
- añadido residual antisimétrico acotado, incertidumbre en ratio y riesgo derivado;
- añadido loss con máscara training-only, NLL física, consistencia y auxiliar aislado;
- añadidos config, ADR, runner fail-closed y nueve tests unitarios;
- Pyright 0, Ruff focalizado limpio y Pytest completo aprobado con 7 skips;
- publicados `cae7d1f`, `c58d07f` y `7945e99` en la rama activa;
- ejecutado gate clean-tree ideal-foreground: Pearson `1.0`, slope `.9999995232`,
  sign `1.0`, oddness `0/0`, translation `0`, rotation-square `.00171029`,
  zero-unknown `1.0`, 336.398 parámetros;
- no se abrió eAP, EvTTC, RGB ni ninguna etiqueta TTC real.

Pendiente inmediato: implementar el generador/runner de aprendizaje event-only
sintético y exigir los mismos gates sobre máscaras predichas held-out. Este resultado
no autoriza real-data, comparación Garl ni claim SOTA.

## 2026-08-02

- auditado el repositorio completo y preservados cambios ajenos;
- corregidos contratos Garl, calibración, timestamps, sampling y métricas firmadas;
- verificada paridad raw/resized y de salida de modelo;
- corregido el pooling BF16 high-resolution;
- implementados padding/máscaras, atención por ventanas, merge 2x2 y mixer temporal;
- rechazado KDA por regresión;
- implementado trainer event-only raw/on-demand sin cache global;
- añadido gradient accumulation, orden determinista por época y perfil full con Git
  limpio;
- añadido runner `screen/full`, seeds 7/13/23, freeze, predict/score EvTTC y
  validación local de submission;
- completado smoke real 16/16; MiD macro `1868,3186`, no promocionable;
- comprobado shard cache 256 y descartado 4.096 por presión de RAM;
- eliminados launchers/aliases rotos, protocolos/manifests duplicados y código
  engañoso;
- eliminados CARLA local, `artifacts/runs` y `artifacts/features`; ~101 GiB
  adicionales recuperados en esta pasada;
- preservados resúmenes compactos de resultados negativos;
- corregidos tests de evidencia para que un clon limpio omita artefactos locales
  opcionales;
- Ruff, Pyright y Pytest completo pasan sin los árboles generados;
- implementada auditoría falsable de shortcut semántico, cinco brazos x tres seeds
  y control frame-varying, sin dataset real;
- demostrado que varianza/VISReg pueden estar sanos estadísticamente y conservar
  el shortcut; R²-lite no supera el gate y queda rechazado;
- residual temporal pasa el shortcut lento, pero falla el control frame-varying:
  se conserva solo como canal `z_delta` candidato junto a `z_level`;
- publicados `7ec2b90`, `cbdf54c` y `7d33989` en la branch activa.

Pendiente prioritario:

1. JEPA high-resolution compatible con salida de nivel y residual;
2. `level` vs `level+temporal_residual` en screen pareado y probes reales;
3. RGB-E causal como ablación separada;
4. inputs EvTTC Tabla VI label-free;
5. full 7/13/23 solo tras mejorar claramente el screen;
6. evaluación oficial eAP/CodaBench.

No implementar R²/HSIC/CMI ni INTACT antes de ese gate: el primero fue rechazado
y el segundo requiere acciones expertas ausentes en este problema.

No existe claim SOTA.
