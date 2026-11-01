# Fusión con soporte nativo explícito: F2 y control de capacidad

Protocolo fijado antes de construir la nueva tabla o entrenar candidatos.
La campaña anterior de reinicios completó 48 ajustes y rechazó IQn: mejoró
el diagnóstico de historia recortada pero empeoró Slider500 limpio. No se
reutilizarán sus resultados oficiales inexistentes ni se abrirá el benchmark.

## Entradas y fuentes

Recuperar diez coordenadas causales de los estados 340 ya existentes:
RGB count/8 en169, eventos count/8 en339; cuatro relojes RGB165:169 y eventos335:339.
Cada reloj representa, en segundos, query-anchor, query-available, intervalo desde
el anchor previo y available-anchor. NO representa duración total de historia.
Orden explícito: RGB[clock4,count/8], eventos[clock4,count/8], es decir
índices165:170y335:340. La tabla incluye una bandera de contexto ejecutado por
modalidad como metadato de auditoría; los diez inputs son los ya indicados.

2531carreteras: unir por token exacto a data_v3/training.npz. Slider500: estados
de las76consultas con fusión en la caché149. Consulta76 conserva la ruta no
disponible: soporte cero explícito, sin inventar un estado340. Verificar paridad
de las coordenadas de carretera en las72consultas reextraídas cuando sea posible.
Conservar razón de ausencia. No derivar soporte desde targets o errores.

Preparar también, sin evaluar nuevos modelos, tablas por token para eAP TRAIN40
activo (88744), Test12 sin etiquetas (6762) y las447variantes de reinicio. Estas
últimas son solo diagnóstico secundario en esta campaña, no aumento de train.
Los campos de geometría/calidad permanecen idénticos. Mantener fuente, ROI,
modelo, reloj y unión de identidades; fallar ante duplicados o fuentes incompatibles.

## Modelos y comparación

IQ congelado como referencia OOF, reproducido bit a bit en native_context_v1.
F2: mismos33inputs de IQ más10soporte, MLP43→18→4,868parámetros entrenables.
Fcap: solo33inputs originales, MLP33→23→4,878parámetros entrenables.
El control tiene10parámetros más (1,15%): no se afirmará igualdad exacta de
capacidad funcional. Sirve para comprobar si aumentar capacidad sin información
nueva explica la mejora. Ambos conservan la fusión inversa y disponibilidad de IQ.
No modificar sus límites, retorno a F0, tratamiento del signo ni normalizadores
de productores. Implementar en un paquete nuevo; no modificar la entrega.

Semillas7,13,21; ocho folds de familia;300updates AdamW;lr0.01;weight_decay0.01;
clip1. Normalización y escala de MAE solo con train limpio. Pérdida igual a QA-IQ:
mitad limpia y mitad con corrupción de un experto en25%de filas, factores
-3,-1,0.25,4. Los indicadores de soporte permanecen asociados a su observación;
la corrupción es un control artificial de fiabilidad, no un cambio físico del sensor.
Datos2608de25grupos elegibles; excluir ocho carreteras ySlider750/1000.48ajustes.

La auditoría previa muestra counts5/4 constantes en carretera y distribuciones
diferentes en TRAIN40OOF/Test12deployment. Esto puede producir una señal de dominio;
no atribuir causalmente una mejora al soporte solo por ver correlación.
EnF2, coordenadas nuevas con varianza de train<1e-12 quedan enmascaradas a cero
tras normalizar, también en inferencia, para no extrapolar pesos sin observaciones
de variación. Guardar la máscara por fold y en refit. Los33inputs originales usan
el contrato normalizador original; Fcap no añade datos nuevos ni esta máscara.

## Selección y alcance

Cada candidato requiere cobertura completa y no empeorar las cuatro métricas
primarias frente a IQ: MAE/RAE macro por grabación en carretera ySlider500.
Si ambos pasan, seleccionar menor media de las cuatro razones candidato/IQ;
desempate deterministaF2. Si solo pasa uno, seleccionarlo. No cambiar la regla.
Reportar todas las semillas. Atribuir ventaja al soporte solo siF2 también
dominaFcap en las cuatro métricas; en otro caso la causa queda sin establecer.

Solo el candidato seleccionado pasa a refit completo y control eAP TRAIN40
activo frente a la entrega congelada (MAE/RAE no empeoran), conservando que es
resustitución y no evidencia independiente. Test12solo mide deriva sin etiquetas.
Ninguna selección local demuestraSOTA ni resuelve el evaluadorSlider750.
Sin candidato admisible: conservar negativos y no reabrir el benchmark.
