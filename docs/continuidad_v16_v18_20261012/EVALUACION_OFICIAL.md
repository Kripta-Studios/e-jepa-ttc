# Preparación oficial del candidato conservado

Estado: **PACKAGED_NOT_SUBMITTED — ENVÍO MANUAL DEL USUARIO**. El candidato no tiene score oficial.

Tu última instrucción es que el agente no lo entregue en CodaBench. El trabajo
termina en preparar y verificar el archivo y explicarte el procedimiento. No
queda ninguna acción automática de subida ni solicitud de permiso programada.

El ZIP original está en [submission/PHASE_ONLY_S7_GarlTTC_submission.zip](submission/PHASE_ONLY_S7_GarlTTC_submission.zip).
Se ha copiado sin modificar bytes y comprobado su SHA-256, CRC, los 6.762 tokens
exactos del INPUTS congelado y cada TTC frente al CSV sellado. La verificación
de las 46 fuentes, padres, normalizadores y cabezas ligados a la congelación no
encontró discrepancias. Véase [INDEPENDENT_VERIFICATION.json](submission/INDEPENDENT_VERIFICATION.json).

Se consultaron las páginas reales de [CodaBench](https://www.codabench.org/competitions/17289/)
el 12 de octubre de 2026, aproximadamente 20:10–20:22 UTC, mediante navegador.
Las capturas textuales se conservan en verification/CODABENCH_*.txt. La primera
respuesta del lector web devolvió contenido ajeno al benchmark; esa respuesta
se descartó al contrastarla con el navegador, que sí mostró GarlTTC Benchmark.

Comprobado en la interfaz:

- La modalidad RGB+eventos es propia del benchmark y la tabla declara ambos usos.
- Se exige un ZIP con exactamente submission.json en la raíz, esquema
  garlttc_prediction_v1 y un TTC en segundos para cada sample_token público.
- El ranking ordena overall_MiD ascendente, con pesos c/s/l/n = 0,5/0,3/0,1/0,1.
- La primera fila pública visible era Garl-TTC, ID 817519, overall_MiD **45,03**,
  MiDc 53,13, MiDs 37,58, MiDl 40,61 y MiDn 31,29; 6.762 muestras.
  Son cifras redondeadas de la interfaz, no precisión completa del scorer.
- La vista no autenticada muestra Login y avisa de que la participación necesita
  aprobación de los organizadores. No acredita si tu cuenta ya está aprobada.
- No se encontraron condiciones explícitas que admitan o prohíban el historial
  causal entre muestras hasta 650 ms. La ausencia de prohibición no equivale a
  confirmación de equivalencia con la pareja temporal nativa del baseline.

El [repositorio oficial](https://github.com/NAIL-HNU/Garl-TTC) también distingue
metadatos test públicos de TTC privados y dirige la evaluación a CodaBench.
La fila del paper no se usa como sustituto del ranking vivo.

Antes de tu envío deben quedar resueltos la participación aprobada de tu cuenta
y el contrato de historial. El paquete general exigía autorización aparte en
[AUTORIZACION_SUBMISSION_OPCIONAL.md](package/AUTORIZACION_SUBMISSION_OPCIONAL.md):
«No se ejecuta por adjuntar el paquete general». No se han aceptado términos,
solicitado registro, enviado mensajes a organizadores ni subido predicciones.

## Cómo hacerlo tú

1. Abre [GarlTTC Benchmark](https://www.codabench.org/competitions/17289/)
   y pulsa **Login**; si no tienes cuenta, **Sign-up**.
2. En **My Submissions**, comprueba tu participación. Si aún no estás inscrito,
   lee y acepta las condiciones y solicita participar. La competición indica
   que los organizadores deben aprobar esa solicitud; espera su aprobación.
3. Antes de subir, confirma con los organizadores que admiten el historial causal
   descrito en la model card. El texto de consulta de abajo está listo para copiar.
4. En la fase test habilitada dentro de **My Submissions**, usa el control de
   subida que aparezca en tu cuenta y selecciona únicamente
   **PHASE_ONLY_S7_GarlTTC_submission.zip**. No selecciones el ZIP de continuidad
   ni el paquete grande de informes. No descomprimas ni vuelvas a comprimir el ZIP.
5. Si solicita nombre de método, indica **PHASE_ONLY_EVENT_S7**. En los campos de
   modalidad/factsheet declara **RGB/Image + Event**, padres congelados e historial
   causal. Usa [MODEL_CARD.md](submission/MODEL_CARD.md) como ficha del método.
6. Confirma tú el envío una sola vez y espera a que termine el evaluador. Si
   falla, conserva el log y revisa el motivo antes de reintentar; no cambies TTC.
7. Guarda ID/recibo, fecha, fase, versión del scorer si se facilita,
   **overall_MiD**, **MiDc/MiDs/MiDl/MiDn**, tasas de fallo y número de muestras
   (**6.762**). Conserva el resultado aunque sea desfavorable.

Los controles posteriores al login no se han inspeccionado en una cuenta
autenticada; por eso no se inventa su nombre exacto. Ningún valor del parser
local sustituye el resultado del evaluador. No generar variantes según test12.

SHA-256 del único archivo que debes subir:

```text
60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1
```

Texto preparado para aclarar el historial, **no enviado**:

> Queremos evaluar un estimador RGB+eventos con pesos fijados antes de test12.
> Cada token utiliza únicamente sensores causales del mismo objeto, hasta ocho
> observaciones y 650 ms de soporte admitido (máximo observado 600.006 µs).
> Los clips solapados se deduplican para formar la historia, manteniendo todos
> los tokens de entrega. No se emplean TTC test ni sensores futuros. ¿Se admite
> este uso de historial entre muestras en la misma tabla, y cómo debe declararse?
