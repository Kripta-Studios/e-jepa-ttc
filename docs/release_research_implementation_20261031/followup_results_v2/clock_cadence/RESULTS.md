# Reloj y cadencia: control de dependencia

Desarrollo expuesto, parámetros fijos, comparación con cada contexto de eventos.

| Cadencia | Lectura | MAE contexto | MAE lectura | Delta CI95 |
|---|---|---:|---:|---|
| original | persistence | 1.947555324 | 3.099852894 | [-0.25305021897951746, 2.716184594472248] |
| original | ewma | 1.947555324 | 1.958482093 | [-0.7931645354137086, 0.5871154207231702] |
| original | transport | 1.947555324 | 3.099852894 | [-0.25305021897951746, 2.716184594472248] |
| original | observer | 1.947555324 | 1.987680041 | [-0.3861329442262649, 0.39023782535812307] |
| canonical | persistence | 1.941890262 | 3.104231759 | [-0.23947804675081305, 2.716184594472248] |
| canonical | ewma | 1.941890262 | 1.952822072 | [-0.7931547058611585, 0.5871205875392543] |
| canonical | transport | 1.941890262 | 3.104231759 | [-0.23947804675081305, 2.716184594472248] |
| canonical | observer | 1.941890262 | 1.981365164 | [-0.3861329442262649, 0.39004321363202316] |

No cambia la cabeza de fusión ni selecciona un nuevo candidato.
Las ausencias y fallbacks se conservan en las predicciones selladas.
