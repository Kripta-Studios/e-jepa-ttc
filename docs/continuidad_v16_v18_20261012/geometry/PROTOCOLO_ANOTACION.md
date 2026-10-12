# Correspondencias independientes pendientes

Se conservan los doce paneles originales y las 72 filas, sin TTC ni errores del
predictor. No se han repetido los ensayos 20/40/80 ms ni alterado su selección.
Completar ANNOTATION_TEMPLATE.csv con coordenadas event nativas 1280×720 y RGB
nativas 1920×1200. El panel derecho está escalado a 1280×720; no copiar sus
coordenadas de pantalla como si fueran las nativas.

Para cada punto, un anotador identifica una correspondencia visible o registra
visibility=ambiguous/absent. No rellenar coordenadas de puntos ausentes. Registrar
identidad del anotador y marcar uncertain. Un segundo anotador revisa los puntos
sin consultar TTC, errores ni decisiones del registro; adjudicar desacuerdos.

La separación por escena se fija ahora, antes de recibir coordenadas:
FCWD1/FCWD2 para desarrollo geométrico y FCWD3 para comprobación. Son ocho y
cuatro paneles respectivamente. La selección previa de paneles incluía aceptados
y rechazados; por tanto la cobertura de esta muestra no estima la cobertura
poblacional sin sesgo. No se modifica silenciosamente esa selección histórica.

No se define un umbral de promoción a posteriori. Antes de probar alternativas,
registrar tolerancia geométrica, criterio de falsos matches, tratamiento de
ambigüedad, cobertura y coste. Medir la proyección en el mismo sistema de
coordenadas de las anotaciones. NCC y aceptación no sustituyen el error de
reproyección. La transformación debe ser coherente durante cada historia.

Con cero correspondencias independientes completas, el error de referencia es
desconocido y ninguna variante puede promocionarse. Una futura variante admitida
obliga a reevaluar todos los comparadores afectados sobre la población completa.
