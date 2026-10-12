# Límite global de réplicas condicionales

Antes de los primeros fits de G/H se registra un máximo de una familia de réplicas condicionales en las colas pendientes. Se conserva la primera admitida según el orden de ejecución G→H. A/B/D eran comparaciones de tres semillas fijadas de antemano; E/F ya han sido rechazadas y no se reabren.

`SELECTED_REPLICATION_FAMILY.json` se congela antes del primer update condicional. Otra cola no puede sustituirla después de observar sus resultados. Los gates originales deben pasar íntegros; ni una mejora parcial ni un intervalo favorable reclaman ese cupo. Tampoco se reasigna el presupuesto de réplicas no ejecutadas a variantes de la misma cola.
