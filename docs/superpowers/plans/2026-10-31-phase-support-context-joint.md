# Interacción entre soporte explícito y reinicios reales del contexto

Protocolo previo a estos ajustes. Las campañas anteriores quedan congeladas:
IQn mejora reinicios pero empeora Slider limpio; F2 con QA mejora Slider y
empeora carretera; Fcap con QA hace lo contrario. Ninguno fue promovido.
El fold Slider de F2/QA solo aprendía variación en el intervalo RGB; no había
variación de los recuentos nativos entre sus carreteras de train.

Hipótesis: el soporte puede ser útil para distinguir una historia recortada de
una completa si ambas se observan durante entrenamiento. Se prueba una única
interacción adicional con las cachés ya selladas, no nuevos hiperparámetros.

Dos condiciones: F2n (43→18→4,868parámetros) y Fcapn (33→23→4,878parámetros).
Reutilizar exactamente NativeSupportRouter de la campaña anterior (modosF2/Fcap),
la conversión física, la fusión inversa y las reglas de ausencia. Paquetes nuevos;
no editar código ni recibos de campañas completadas.

Pérdida0.5limpia+0.5reinicios reales, con las447variantes y las2608consultas.
Excluir de cada fold TODOS los reinicios de sus grupos reservados. Equilibrar
familias y consultas, dividir el peso de una consulta entre variantes admitidas.
Sin corrupción multiplicativa adicional: este término reemplaza al deQA, igual
queIQn. La escala deMAE y normalización de33inputs originales usan train limpio.

ParaF2n, normalizar las10coordenadas nuevas y calcular su máscara de varianza
con la unión de soportes limpios y aumentados ADMITIDOS DE TRAIN, nunca validación.
Esto difiere expresamente deF2/QA: ahora existen variaciones de soporte vistas
en train. Umbral de varianza1e-12; clips y resto del normalizador permanecen iguales.
Guardar máscara, centro, escala y hashes de las identidades usadas por fold.
ParaFcapn, no entran las10variables nuevas ni su máscara.

Tres semillas7,13,21; ocho folds por familia;300updates AdamW;lr0.01;
weight_decay0.01;clip1;48ajustes. No búsqueda adicional según resultados.
Referencias OOF selladas: IQ del control reproducido, IQn de native_context_v1
y F2/Fcap de native_support/campaign_v1, unidas por token. Solo IQ selecciona.

Cada candidato requiere las cuatro métricas limpias no peores queIQ y cobertura
completa: MAE/RAE macro por grabación de carretera ySlider500. Si ambos pasan,
seleccionar menor media de cuatro razones candidato/IQ; desempateF2n.
Reportar cada semilla y reinicios OOF como diagnóstico secundario sin selección.
No afirmar beneficio específico del soporte salvo dominancia frenteFcapn en
las cuatro métricas (no peor en todas y mejor en al menos una); aun así es
asociación experimental y no prueba de generalización independiente.

Solo un seleccionado pasa a refit y conservación activa eAP TRAIN40 MAE/RAE
frente a la entrega congelada; señalar resustitución y estados upstream OOF.
Test12sin etiquetas no certifica precisión. No abrir benchmark para escoger
variantes. Todos los resultados y fallos se conservan; no se declaraSOTA.
