# Captura PAIR de despliegue y conservación nativa

El replay original de contextos ya terminó; no se repite como sustituto de
productores. Falta PAIR133 de las observaciones de despliegue. Se capturará una
vez cada observación física original con sus productores sellados, lotes de uno,
historia completa y el mismo decodificador. Las entradas públicas verificadas
contienen 5.805 observaciones por modalidad y 6.762 aliases de consulta.

Primero se perfila un prefijo de dos consultas físicas en CPU y un hilo; solo
ejecuta el original. Se guardan PAIR133, delta, fase sin clipping, PHASE17 y
expert3, con identidad propia. No se abren targets Test12. La extracción completa
espera el resultado F0 de population2608 y un recurso libre; no duplica su worker.

Se verifica el SHA de cada tensor de eventos al usarlo y cada shard RGB antes de
decodificarlo, contra el inventario público sellado. Las fuentes y pesos se
vuelven a verificar al cerrar. La preparación no cambia resolución, recorte,
cadencia ni política de cold start. El delta nominal de eventos se conserva
separado de los timestamps físicos; no se modifica para forzar una comparación.

Antes del candidato, reconstruir el original y compararlo con los diagnósticos
históricos: igualdad categórica, rtol=1e-5/atol=1e-6 en tensores y
rtol=1e-4/atol=1e-4 en TTC. Un fallo se conserva y se investiga; no se amplían las
tolerancias tras observarlo. La comparación causal del candidato es pareada en
la misma plataforma, con los mismos lotes y pesos, reconstruyendo PHASE17,
contextos y cabeza. La ruta de identidad de eventos debe permanecer exacta.

No afirmar conservación de precisión nativa sin evaluación autorizada. Una
paridad de predicciones o un cambio pequeño son resultados distintos de medir
Overall/Crucial/Negative. No se publica ni sustituye el release desde estos scripts.
