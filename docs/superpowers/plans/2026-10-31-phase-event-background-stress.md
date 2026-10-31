# Prueba fija de contaminación por fondo

El experto de expansión temporal NCC recupera movimiento en contornos ideales,
pero sus resultados reales no mejoran la versión congelada. Una hipótesis física
pendiente es la mezcla de movimientos dentro de la ROI. Esta prueba diagnostica
el estimador existente; no modifica sus parámetros ni selecciona un nuevo modelo.

Generar expansión frontal ideal con las semillas nuevas 61 y 73 y 4096 eventos
de objeto. Añadir un contorno estático independiente con proporciones de eventos
de fondo 0, 0.25, 0.50 y 0.75 respecto al total. Mantener exactamente los eventos
originales del objeto, el reloj, la ROI y el TTC; el fondo ocupa un rectángulo mayor
dentro de la misma ROI. Es un proceso de contornos para sensibilidad, no un
simulador fotométrico ni una atribución de los fallos de Slider a esta causa.

Ejecutar el solver CPU congelado endpoint/temporal_ncc, mismo presupuesto y gates.
Guardar código, hashes de eventos, parámetros verdaderos, decisiones y error.
Contar abstención como fallo de recuperación, y distinguirla de una estimación
finita equivocada. No abrir sensores ni etiquetas reales. Si la contaminación
degrada el resultado, estudiar después un criterio de soporte de movimiento;
esta prueba por sí sola no justifica recortar la ROI de la entrega ni entrenar
con las consultas oficiales. Mantener los ocho casos, incluidos negativos.
