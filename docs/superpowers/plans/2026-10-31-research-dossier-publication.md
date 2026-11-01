# Dossier de investigación y publicación en GitHub

Solicitud: redactar Markdown detallado de lo realizado durante los últimos días,
incluyendo resultados, implementaciones, pruebas y fallos, y hacer add/commit/push
de todo el trabajo a GitHub. Esta solicitud autoriza la publicación que las
iteraciones anteriores habían dejado local.

Arquitectura documental: dossier en `docs/research_log_20261029_31/` con índice,
cronología, protocolo, modelos, resultados, rendimiento, verificación, entrega y
límites. Los datos numéricos proceden de recibos JSON, XML y commits existentes.
Se preservan los documentos anteriores como fotografías de cada iteración.

- [x] Inventariar commits, artefactos, evidencia y archivos pendientes; comprobar
  identidad de los ZIP y delimitar el material publicable del proyecto.
- [x] Redactar el dossier, copiar evidencia final pequeña que todavía era local,
  enlazarlo desde README y registrar los paquetes con tamaño, SHA-256 y función.
- [x] Validar enlaces locales, cifras/tablas y diff; una revisión independiente
  de documentación/evidencia, resolver problemas importantes, commit y push de
  la rama actual; publicar los ZIP propios como adjuntos de una prerelease.
- [x] Verificar commit remoto y los hashes de los adjuntos publicados; registrar
  los recibos de publicación y subir el cierre documental.

Reglas: no subir sensores crudos, entornos, cachés, código fuente de artículos
ajenos ni carpetas temporales de agentes. Se versionan los Markdown recibidos y
el plan extraído; los ZIP del proyecto se publican como assets, evitando blobs
grandes en Git. No se fuerza un push ni se modifica main. No se envía correo a
los organizadores. Los resultados locales no se rebautizan como oficiales ni
como SOTA. No se vuelve a entrenar ni a ejecutar una batería de 15 minutos por
un cambio documental: las verificaciones nuevas corresponden al contenido,
enlaces, artefactos y publicación.

Ruling: «todo a gh» comprende código, documentación, evidencia y entregas propias;
los archivos de trabajo no publicables permanecen locales. Los binarios van a
GitHub Releases y el repositorio conserva su inventario verificable. Si un
artefacto no cabe o falla su hash, no se sustituye silenciosamente.

Resultado: dossier de diez Markdown principales y notas de release (~11.400 palabras), treinta enlaces locales verificados, trece filas de métricas exactas, diez ZIP (200.240.822 bytes) y SHA256SUMS publicados. Revisión independiente: cero Critical, una Important corregida (micro frente a macro), una Minor de redacción conservada en REVIEW.json. Primera publicación53ffc341; tag fijado; PUBLICATION.json verifica onceassets por tamaño y digestGitHub. El commitde recibos se sube sin mover eltag. Sin cambios en código/modelo; no se repitió entrenamiento ni suite completa.
