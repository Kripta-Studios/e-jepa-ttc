# 07 — Fuentes y alcance

## Fuentes del usuario

Base V12: https://github.com/Kripta-Studios/e-jepa-ttc/tree/1a686e338a1145c6794bec0704bdae4cda95f06d

Carpeta examinada: `docs/agent_handoff_20261010`, índice y ocho informes01–08, tablas y predicciones referenciadas. Base V13: https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0

Los recibos `SOURCE_PINS.json` y `GIT_TREE_RECONSTRUCTION.json` fijan código y árboles. El ZIP original tiene SHA-256 `20ce464c970d4ade5e0c41ecdde46e712ee52b399f4e4941a991b2aefd26524e`,46.007.562 bytes y6.756 miembros. Sin credenciales, raw ni grandes checkpoints en esta revisión.

## Verificado aquí

Integridad del ZIP, seguridad de extracción,6.755 hashes; manifiesto del handoff511 archivos / 87 enlaces. Reconstrucción del rootGit de todos los blobs de ambos snapshots y comparación con la API pública. ReplayFCWD597 × 4 filas con diferencia máxima de MiD≈1,7053e-13 por aritmética flotante. Nuevo análisis descriptivo desde CSV nativo, con sus hashes. Pruebas CPU del código de pesos/loss contra fase/modelo nativos mediante importación aislada. Verificación del patch en un repositorio de ensayo.

No se ha completado clone con historial: DNS impidió el comando. Los snapshots comprometidos sí se inspeccionaron completos por acceso programático; eso no equivale a revisión manual de cada una de sus miles de líneas. Se leyeron íntegramente los informes y los módulos relevantes al nuevo estudio.

## No verificado aquí

Inferencia nueva de los productores, datasets crudos, reentrenamiento, pruebas completas del repositorio ni estado vivo V13. PowerShell/Windows, CUDA, telemetría térmica y runtime integrado no ejecutados. Las nuevas funciones son referencia testeada, no un trainer ya integrado. Ninguna prueba sintética acredita una mejora real de TTC.

## Investigación pública contextual

- Garl release oficial: https://github.com/NAIL-HNU/Garl-TTC — labelsTRAIN, metadataTEST y submission para scoring oficial.
- eAP/Garl: https://arxiv.org/html/2603.16303v1 — definición de tarea, métricas y modalidades; no mezclar sus números con transferencias locales.
- FCWD: https://nail-hnu.github.io/EventAidedTTC/ — tres secuencias y archivos de calibración; https://github.com/NAIL-HNU/event_aided_ttc — referenciaMATLABR2022a.
- CodexCLI: https://developers.openai.com/codex/cli/reference/ — `codex --cd` abre el workspace indicado. No se requiere cambiar permisos globales ni modelos del cliente.

Estas fuentes contextualizan un protocolo; no sustituyen los resultados del usuario. Los nuevos coeficientes/gates son propuestas prospectivas, no hechos extraídos de un paper ni resultados ya obtenidos.
