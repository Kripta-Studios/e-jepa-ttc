# Ampliación contable de V18: 16 de octubre

El techo acumulado de scopes propios pasa de **172.800 a 216.000 segundos**
(48 a 60 horas). Se aplicó a las 13:59:15 de Madrid, al liberar el hijo de B1
el lock existente `V18_GPU.lock`. No se terminó ningún proceso ni se reinició
un checkpoint. El coordinador conservó su PID y abrió el siguiente segmento.

La autorización es el mensaje previo del usuario: «sí tienes permiso para
continuar y para ampliar el presupuesto de todo, no pares hasta acabar, y vete
revisando cada hora en punto que todo se esté ejecutando y vaya bien».
El nuevo techo de 60 horas es una ampliación acotada bajo esa autorización;
no es una obligación de consumir las horas restantes.

## Motivo y alcance

A las 13:47 quedaban 6.779 updates B1. El ritmo observado entre segmentos
durables era 0,614 updates/s: aproximadamente 11.034 segundos de entrenamiento,
además de evaluación y exports. Quedaban 8.619 segundos bajo el techo anterior,
incluida una reserva activa. El margen ya no cubría la cola admitida.

Se conservan el millón de updates físicos, las reservas técnicas/de recuperación,
los límites de memoria/disco, los cuatro experimentos adaptativos como máximo,
las condiciones de las ramas, los splits y los gates. La ampliación no registra
una nueva pérdida, sweep, familia de réplica o refit. No modifica ningún fichero
científico congelado ni las recetas de los endpoints.

## Conservación del historial

Los **284 asientos** existentes del ledger permanecieron idénticos. En la frontera
había **164.154,361 segundos contabilizados o reservados**. La cifra es menor que
la reserva observada a las 13:47 porque terminó y se liquidó ese scope; no hubo
reset ni descuento de asientos históricos.

El recibo está en
`artifacts/joint_measurement_20261012/verification/BUDGET_AMENDMENT_20261016/RESULT.json`.
En el mismo directorio se preservan los bytes originales de `LEDGER_BEFORE.json`
y `AUTHORIZATION_BEFORE.json`, junto con `LEDGER_AFTER.json`.

`BUDGET_AUTHORIZATION_20261016.json` registra la ampliación. Por compatibilidad con
el runner ya congelado, `BUDGET_AUTHORIZATION_20261014.json` actúa como documento
activo y contiene la misma autorización vigente. Su versión anterior se conserva
sin cambios en el archivo indicado arriba; no se reinterpreta como una autorización
de 60 horas emitida el día 14. El código exige que el techo activo coincida con
el ledger, y ambos se actualizaron bajo el mismo lock después de la liquidación.

La configuración científica primaria conserva su valor histórico de 64.800 s
y su hash original. El techo efectivo del runtime está en el ledger y en la
autorización aplicada; modificar aquella configuración invalidaría su genealogía.

[Recibo de aplicación](evidence_budget_20261016/RESULT.json) ·
[Autorización vigente](evidence_budget_20261016/AUTHORIZATION_APPLIED.json) ·
[Autorización anterior preservada](evidence_budget_20261016/AUTHORIZATION_BEFORE.json)
