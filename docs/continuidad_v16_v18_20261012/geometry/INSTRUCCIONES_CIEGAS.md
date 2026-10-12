# Anotación de correspondencias

Abre los doce PNG a resolución original. Cada archivo mide 2560×770: la cabecera
ocupa 50 filas; eventos a la izquierda (1280×720), RGB a la derecha mostrado a
1280×720 pero procedente de una imagen nativa 1920×1200.

Identifica hasta seis puntos inequívocamente compartidos por ambas vistas en
cada panel. Usa esquinas o intersecciones precisas de bordes; evita superficies
uniformes. Completa `ANNOTATION_TEMPLATE.csv` con coordenadas nativas, identidad
del anotador y `visibility=visible`. Si no se puede identificar un punto, usa
`ambiguous` o `absent`, explica el motivo en `adjudication` y deja sus coordenadas
en blanco. Marca `uncertain` cuando corresponda; no inventes seis puntos para
cumplir una cuota.

No consultes TTC, errores del predictor ni los resultados de los algoritmos de
registro. Un segundo anotador debe revisar de forma independiente los puntos y
registrar los desacuerdos/adjudicación. No cambies `scene_role`.

Si tu editor informa coordenadas de centros de píxel del PNG completo, con
origen cero y sin escala de zoom, convierte así:

```text
Evento: x_native = x_png; y_native = y_png - 50
RGB:    x_native = (x_png - 1280 + 0.5) * 1920/1280 - 0.5
        y_native = (y_png - 50 + 0.5) * 1200/720 - 0.5
```

No utilices coordenadas de una captura del navegador escalada. Si empleas una
convención continua de bordes en vez de centros de píxel, documenta esa convención
y mantenla coherente al medir reproyección. Devuelve el CSV y el identificador
del paquete de imágenes utilizado.
