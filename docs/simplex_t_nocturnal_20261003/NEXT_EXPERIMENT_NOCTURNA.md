# Única propuesta posterior: coste integral con modelos congelados

Estado: NO EJECUTADA. Presupuesto de nuevos fits y updates: cero.

La pregunta es cuánto del ahorro aparente de cabeza sobrevive al preparar el contexto del ROI actual y ejecutar los productores congelados. Comparar los nueve modelos ya perfilados sobre las mismas64 consultas TRAIN elegidas por hash, sin targets ni selección por error. Conservar pesos, normalizadores, PHASE17 y recetas; no introducir estado recurrente persistente, caching gratuito entre ROI distintos ni nuevos modelos.

Medir por etapas raw slice/ROI, expertos, features, cabeza y emisión TTC; batch1, cache warm y cold de aplicación separados, sin vaciar cachés globales del SO. Añadir bloques emparejados de cabeza en la misma sesión para caracterizar la variación de carga observada; registrar todas las repeticiones, no escoger la más rápida. A5+PAIR comparte A5 y no cuenta como dos encoders independientes. Auditar que el productor excluido no sea llamado ni determine historia/validez; si la genealogía heredada impide una ruta independiente, declarar esa limitación sin cambiar consultas.

Dependencia exacta: un slot exclusivo explícito GPU/lectura pesada concedido por el propietario de Stage70–76 en la interfaz de coordinación, más fuentes TRAIN autorizadas y límites activos. La interfaz actual no concede ese slot. No detener Stage70–76 ni inferir permiso de GPU aparentemente libre. No abrir Stage76, public validation, private test, EvTTC test o CodaBench.

Los cribados de precisión actuales permanecen negativos para las reducciones y agregadores. Medir coste no cambia sus intervalos ni demuestra no inferioridad. Esta propuesta no autoriza entrenamiento, promoción de H16 o sustitución del H8 registrado. No se ejecuta automáticamente tras la entrega.
