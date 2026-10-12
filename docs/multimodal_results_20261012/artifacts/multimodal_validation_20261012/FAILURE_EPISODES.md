# Episodios trazados

Selección externa descriptiva, ya expuesta; no se interpreta como validación independiente. Los 32 controles H se seleccionaron por SHA-256 del token y estrato, antes de inspeccionar errores. Consultar `H_CONTROLS_SEALED.json` y `h_raw_queries/`.

En `CCRs-1-medium-100-overlap-100:01`, E_CTX parte de fase +0,003510 y añade +0,009249 (TTC +7,8875 s). R_CTX parte de +0,002844 y añade −0,014790 (TTC −8,3213 s). F_TRUE usa el anchor event +0,003510 y añade −0,004199: fase raw −0,000689, fase emitida aproximadamente −0,001665 y TTC −60 s. El signo malo RGB precede a la fusión; la fusión acerca todavía más su propia fase al origen. No es correcto atribuir todo el fallo únicamente al productor RGB.

En `CCRs-1-low-100-overlap-100:24`, E_CTX emite −5,8340 s y R_CTX +1,7266 s. F_TRUE parte de −0,001859, añade +0,006284 y queda en +0,004425: TTC +22,6493 s. El GT publicado es aproximadamente 0,944 s; la fusión amplifica el error respecto al RGB refinado. En la consulta vecina `:23`, F_TRUE emite −39,4919 s y R_CTX +3,5199 s; las vecinas no son escenas independientes.

La igualdad bit a bit se comprueba entre el mismo modelo CPU con y sin hooks; los hooks devuelven None y se eliminan en finally. Las salidas históricas GPU se comparan aparte. Los diagnósticos registran tensores privados de entrada/salida, no alteran parámetros ni actualizan estados de entrenamiento.

La fase cercana a cero explica aritméticamente estos TTC extremos. No demuestra que un residual más pequeño, escoger el TTC mínimo positivo o una nueva compuerta mejore la población. El siguiente experimento debe preservar el buen punto de partida R_CTX y contrastar información event real frente a control nulo de igual capacidad, con identidad y autorización nuevas.
