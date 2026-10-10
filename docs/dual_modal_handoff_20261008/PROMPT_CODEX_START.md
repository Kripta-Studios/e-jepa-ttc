# Autorización nueva — ejecutar V13 DUAL-TTC

Continúa la investigación E-JEPA-TTC para obtener dos modelos: event-only en inferencia
y RGB+eventos. Esta orden activa el paquete; no es sólo una solicitud de documentación.
Implementa, valida, entrena, evalúa y entrega todas las ramas habilitadas por el protocolo.
No prometas SOTA ni declares entrenamiento por existir código o configuraciones.

Lee primero:
- docs/dual_modal_handoff_20261008/README.md
- docs/dual_modal_handoff_20261008/SOURCE_PIN.json
- docs/dual_modal_handoff_20261008/execution_policy.json
- los siete documentos de docs/ y configs/campaign.json del paquete.

Base revisada: b46d40bf1f38ae3e3ea201d58b79391fa475ca3c.
Código científico de la comparación histórica: f13f018421afe5c99104475400f8ffbbefcc434d.
La base no es una orden de reset. Conserva avances locales/ajenos y documenta el HEAD
real. Usa la rama nueva scientific-recovery-v13-dual-modal-ttc y un output root V13.

## Lo cerrado y lo pendiente

No repitas T6, H8/H16/WIDE, TRAIN40 ni las comparaciones EvTTC ya publicadas. E3 nativo
Garl quedó pausado por cambio de estrategia del usuario; no lo reinicies por inercia.
R1 estaba parcial en el snapshot: consulta el propietario vivo y los recibos actuales.
No dupliques su supervisor ni destruyas fragmentos. R1 no es un gate de precisión de V13.
Stage70–76 permanece separado. No mates procesos ajenos ni modifiques el sistema global.

## Cola autorizada

M0: admisión de fuentes y roles, auditoría descriptiva nativa/common-cap60 de los CSV,
paridad de RGB/tiempos/crops, preparación cacheada y pruebas reales TRAIN acotadas.

B: tres puentes sobre H8 congelado (RGB real, frame actual repetido, bloque RGB cero),
2.500 updates cada uno. Dos calibradores diagnósticos de fase de Garl público, 2.500
cada uno. TRAIN40 y transferencia de desarrollo ya expuesta; no tratar como holdout.
Si falla el puente, continúa D: son hipótesis distintas.

D: EV_DIRECT_P2, EV_CORR_P2 y EV_CORR_T4, 24.000 updates cada uno. Se entrenan encoders
ResNet50/FPN y readouts propios. La fuente inicial es genérica ImageNet declarada,
no task checkpoints TRAIN40 ni Garl público que hayan visto grupos DEV.

R: nueva rama RGB entrenable sobre el E seleccionado. R_TRUE y R_REPEAT, 12.000 cada
una. Fusión tardía de evidencia temporal; E congelado y fallback exacto sin RGB.

Réplicas/KD/final: sólo conforme al árbol y gates de docs/04_EXPERIMENTS.md.
No inventes brazos, sweeps, seeds, cambios de loss o entrenamiento extra para rescatar
un resultado negativo. La reserva de una rama no ejecutada no se reasigna.

Techo nuevo: 286.500 updates científicos; 500 técnicos; 13.000 de recuperación.
Máximo físico 300.000. No procede del saldo de la campaña antigua.

## Fronteras científicas que debes respetar

- TRAIN40 completo es entrenamiento de sus propios pesos, no test.
- EvTTC actual ya se utilizó en desarrollo. Su lectura autorizada no reinicia un
  supuesto test ciego; no uses sus scores para ajustar modelos D o elegir seeds.
- Una partición D de aproximadamente 80/20 se agrupa por adquisición y se fija sin
  targets. Sus modelos y teachers deben excluir DEV desde el principio. No puedes
  cargar TRAIN40 para luego llamar holdout a ocho de aquellas secuencias.
- Ancestría unknown limita claims de Garl público, no obliga a parar ramas independientes.
- Event-only de inferencia puede haber usado RGB TRAIN; decláralo, y evalúa por separado
  E sin distilación RGB emparejada y E_KD si existe. No llames a ambos entrenamiento puro E.
- Preserva MiD/MAE/RMSE/mediana/signo/colas. Cap60 común es diagnóstico adicional, no
  modificación de la publicación ni excusa para excluir fallos.
- La primaria de arquitectura es el contrato P2 nativo. T4 usa más contexto y sólo
  entra en la ruta oficial si las reglas públicas admiten esa información.

## Implementación real

La referencia PyTorch del paquete no es un trainer eAP listo. Integra los operadores
con lectores y matrices reales. No conectes el viejo ObjectEventRGBFusion a H8 sin
adaptar su contrato. PHASE17 y los checkpoints históricos conservan sus bytes.

Verifica la selección RGB por timestamp de imagen, disponibilidad real y el crop
específico de cada sensor. No uses tiempos de cajas como tiempos de frame, shift eAP
sobre EvTTC, GT-depth para fusión ni imágenes futuras para rellenar historial.
El número20 de planos nativos no se reinterpreta como10×2 polaridades sin leer la
implementación Garl; conserva la función native_feature y sus fronteras exactas.

Reutiliza durable_io/checkpoints, procesos persistentes, caches acotadas y optimizaciones
ya admitidas. No multipliques controladores con monkeypatches globales. Cada fit conserva
su source/config/data/parent hashes, contador, RNG, sampler y full checkpoint cada100.
Congela todos los endpoints de una ronda antes de observar su comparación D.

Recursos locales: techo agregado23GB, hostmínimo2GiB, recuperación3GiB, discoemergencia10GB
tras reservas. Un trainer pesado nuevo por defecto; otros trabajos sólo si hay slot y
presupuesto agregado. No Runpod, cambios de driver, paging o Control Center.

No te detengas tras generar comandos o compilar. Después de QA/freeze ejecuta la cola
habilitada y monitorízala mediante trabajo durable. Ante recurso transitorio, preserva
estado y continúa tareas independientes. Si todo está bloqueado, entrega la causa
específica, el estado y el comando real de resume, sin renombrarlo negativo científico.

## Entrega y benchmark

Produce los artefactos enumerados en docs/06_FILE_BY_FILE.md. Entrega pesos de los modelos
nuevos o inventario explícito cuando no se incluyan, predicciones, curvas, comparadores
sin cambios, análisis paired/grupos, contabilidad, pruebas, modelcardsE/ER y ZIP+SHA256.

Después de selección D y refit TRAIN40, autoriza export LOCAL sobre inputs públicos de
test disponibles sin labels, sólo con contrato admitido y endpoints ya congelados.
No hagas submissions, no abras labels privados, Stage76 ni puntuaciones selladas.
No push/merge automático. Prepara comandos/schema de submission para revisión humana.
No elijas un nuevo checkpoint tras mirar el leaderboard.

Queremos el modelo más fuerte que se pueda demostrar con evidencia, incluso si resulta
ser una variante derivada de Garl. Si la propuesta pierde, conserva el mejor modelo
existente y entrega la comparación: no sostengas una arquitectura por preferencia.
