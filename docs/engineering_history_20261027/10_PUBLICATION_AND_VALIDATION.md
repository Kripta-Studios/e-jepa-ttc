# Publicación y validación del trabajo acumulado

## Alcance

Esta publicación reúne el trabajo pendiente desde `771c96a23784e1f3955cec21ba1f4d9a063a9145`, junto con una reconstrucción documental del historial iniciado en junio de 2026. Incluye código operativo, pruebas, configuraciones, informes y resultados compactos. No es sólo una edición de documentación: cinco fuentes existentes contienen cambios de runtime, pérdida y transporte, y se añaden las colas y herramientas utilizadas durante el refit y la migración.

La identidad del código histórico V18 sigue ligada a su commit y bundle originales. El árbol actual no se presenta como si hubiera producido retroactivamente todos los experimentos previos. Los bindings y recibos de cada campaña determinan sus fuentes efectivas.

## Protección de identidad de bytes

La revisión previa encontró 39 archivos pendientes con CRLF. La regla general de Git convierte texto a LF, lo que habría cambiado hashes registrados aunque la semántica Python o JSON no cambiase. Se añadieron excepciones de ruta exacta en `.gitattributes` para conservar esas representaciones. El catálogo público de evidencia también conserva sus bytes mediante una regla localizada.

Tres referencias directas hacían especialmente visible el problema: el runner final de Garl, la cola final Runpod y el recibo de revisión de rotaciones anidado. La publicación debe conservar sus bytes ejecutados; recalcular un hash tras reformatear y sustituirlo en el informe no equivaldría a conservar el experimento.

Los tests nuevos corregidos para portabilidad no forman parte de la identidad de los pesos. La corrección de un fixture no se registra como entrenamiento, inferencia ni validación científica ejecutada de nuevo.

## Comprobaciones realizadas

La auditoría inicial examinó 677 archivos pendientes, aproximadamente 9,29 MB, antes de añadir estos capítulos. No encontró credenciales reales en las búsquedas específicas y la revisión de candidatos; esto no es una prueba matemática de ausencia. Los `sample_token` del benchmark son identificadores de muestras, no secretos de autenticación. El mayor archivo era una tabla de aproximadamente 2,06 MB.

La validación focalizada cubrió 128 archivos de tests relacionados con runtime, transporte, reanudación, OOF, colas, exportación, Test12, migración y transferencia. La primera ejecución produjo **864 aprobados, ocho fallos y uno omitido**. Los ocho fallos procedían de dos fixtures desactualizados: una lista de fuentes incompleta frente al cierre de 30 archivos requerido por el export, y una fecha límite fija que ya había expirado. Se corrigieron los fixtures y la ejecución de las pruebas afectadas terminó con **22/22 aprobadas**.

Además se sustituyeron dependencias de artefactos ignorados por datos sintéticos en pruebas de contexto RGB V8/V9 y del archivo de retorno. Esas pruebas deben poder ejecutarse en un clon limpio. No se repitió la campaña científica para corregir la infraestructura de tests.

El análisis estático Pyright terminó sin errores al resolver el entorno instalado mediante `--venvpath ../e-jepa-ttc`. Su configuración global cubre sólo cinco archivos heredados: **no implica que todo el código nuevo esté tipado sin errores**. El intento inicial sin resolver ese entorno produjo 12 errores de resolución de tipos Torch. Los 80 scripts PowerShell se analizaron sin errores sintácticos.

La comprobación del contenido preparado para Git confirmó los 39 hashes de archivos CRLF originales y los 17 hashes del catálogo público. Se analizaron sintácticamente 441 archivos Python pendientes sin errores. Los enlaces locales de los diez capítulos, índice y catálogo existen en el archivo de trabajo, con la distinción de disponibilidad pública indicada en el README.

## Deuda de calidad que se publica explícitamente

La revisión global Ruff detectó **495 hallazgos** en scripts y fuentes operativas, y su revisión inicial de formato cambiaría 148 archivos. Son resultados distintos de la validación funcional focalizada. No se declara que la CI global esté verde ni se desactiva ampliamente Ruff para aparentarlo.

Tras corregir los fixtures quedaron 144 archivos pendientes de formato. `git diff --cached --check` también señala espacios generados en un SVG histórico y líneas finales vacías en seis archivos; se conservan las fuentes y recibos históricos en lugar de alterar sus bytes por estética. Estas advertencias de publicación no se presentan como una comprobación aprobada.

Se evita reformatear en masa fuentes ligadas por SHA a ejecuciones ya terminadas. Una limpieza posterior deberá separar copias históricas inmutables de puntos de entrada mantenidos, y registrar cualquier nuevo binding si cambia una fuente científica. La preservación de hashes no convierte los avisos en resueltos.

Algunas herramientas de transferencia fueron scripts de una operación concreta y dependen de rutas locales, endpoints ya históricos o un adaptador RPC no incluido en Git. Son evidencia del procedimiento; no constituyen por sí solas una instalación portátil de Runpod. Los puntos de entrada científicos reutilizables y los pasos de reproducción se describen en los capítulos de entrenamiento, migración y exportación.

## Qué se incluye y qué permanece fuera

Se publican fuentes, tests, documentación, tablas y recibos revisados. No se fuerza la inclusión de datos crudos, checkpoints, caches, entornos virtuales, logs masivos ni credenciales. El archivo local de preservación conserva los resultados grandes con sus inventarios. Los hashes públicos permiten comprobar identidad; no sustituyen los archivos de modelo necesarios para reproducir inferencia.

El commit y push solicitados publican el repositorio. No envían predicciones a CodaBench, no reinician entrenamientos y no crean ni eliminan recursos de pago. La verificación de la rama remota se realiza después del push; su resultado pertenece al recibo operativo de publicación y a la respuesta de entrega, no se inventa anticipadamente en este documento.
