# Cierre local SIMPLEX-T

Se completaron los 13 checkpoints T6 y la baseline EWMA registrada en tres folds. Esta sesión ejecutó cero actualizaciones de optimizador y no abrió confirmación.

## Identidades y ejecución

Scientific code: `0c1a7285b6b5af869b0bf5a13a9629f08997e7a2`. Freeze SHA-256: `2eb3fff9ea24145c4af16c0c8645eb65893144580e41880790e169239fa8f3a7`. Worktree: `C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v10-simplex-t-companion`. Rama: `scientific-recovery-v10-simplex-t-companion`. HEAD/commit de entrega: `215768633b2dfa443552b0a2e24462211e1a1c59`. El entrenamiento conserva su identidad científica histórica; el commit operativo no lo modifica retroactivamente. La entrega canónica previa fue verificada en su HEAD histórico y queda preservada en canonical_delivery/.

Verificación canónica: `DELIVERY_REVERIFIED_AGAINST_BOUND_SCIENTIFIC_EVIDENCE`. Los 72 endpoints2500 y sus hashes están reconciliados; total científico guardado 180.000 updates. La contabilidad física histórica conserva el intervalo [181105, 181325] incluyendo técnica/incertidumbre de terminaciones previas, sin consumo adicional aquí. El techo no autorizaba más trabajo.

La política local fue enmendada por el usuario a 2 GiB de RAM disponible y 10 GB decimales de emergencia tras reservas; permanece techo RSS del árbol 4 GiB. No se alteró Stage70–76. Se conservan los cambios locales ajenos registrados en [GIT_STATE.json](GIT_STATE.json).

## Reparación y paridad

Se conservaron los seis checkpoints iniciales y los fragmentos posteriores válidos. Los cuatro grupos pendientes usan 33 fragmentos cada uno: 8.195 intentos publicados T2, 8.192 aceptados, tres rechazados. No hubo nueva selección de draws ni cambio de población, seeds, pesos, operaciones numéricas o intervalos. Se repararon la publicación atómica Parquet en Windows, la comparación JSON tupla/lista y la declaración incompleta de recibos T2/T4. El recibo T2 de una invocación fallida no acredita picos o tiempos exhaustivos de entrenamiento.

[PARITY.json](verification/PARITY.json) y los XML guardan la equivalencia bit a bit entre ejecución continua e interrumpida, frente a la rutina congelada; para T2 publicado se conserva tolerancia preregistrada absoluta 1e-10. También se probaron hash cambiado, raíz ausente, Windows1455, journal cortado y reemplazo interrumpido. Los tests no entrenaron modelos.

## Números reconciliados

Cohorte OLD_DEV: 8.192 queries, nueve secuencias, tres folds. Los CSV físicos y evidence.json se verificaron a precisión completa, con diferencia absoluta máxima permitida 1e-10; el markdown redondeado no se usó como oráculo. La tabla siguiente se regenera desde [METRICS.csv](supplement/METRICS.csv).

| Brazo | MiD OLD_DEV |
|---|---:|
| TPR-D0-H1-C64@7 | 137.40793338286844 |
| TPR-D0-H1-C160@7 | 139.87523036235734 |
| TPR-D0-H8-C64@7 | 135.35941009423107 |
| TPR-D0-H8-C160@7 | 136.96098632029526 |
| TPR-D1-H1-C64@7 | 134.61892846963258 |
| TPR-D1-H1-C160@7 | 132.83161914274606 |
| TPR-D1-H8-C64@7 | 125.47750398809649 |
| TPR-D1-H8-C160@7 | 121.64633225755644 |
| PAST_REVERSED-D1-H8-C160@7 | 121.2248640871761 |
| REPEAT_CURRENT-D1-H8-C160@7 | 131.21119698206098 |
| SELECTOR-D1-H8-C160@7 | 143.50373023995019 |
| FREE-D1-H8-C160@7 | 121.37467574825402 |
| TPR-DENSE_OLD-H8-C160@7 | 131.50486888668041 |
| TPR-DIVERSE_MATCHED-H8-C160@7 | 122.00733371090608 |
| TPR-D1-H4-C160@7 | 127.17936209238624 |
| TPR-D1-H16-C160@7 | 118.8657389102478 |
| TRANSFORMER-D1-H8-C128@7 | 126.8146846512566 |
| LATENT-D1-H1-C160@7 | 171.05733318368743 |
| LATENT-D1-H8-C160@7 | 184.07229955443705 |
| LATENT_ZERO-D1-H8-C160@7 | 120.9104157782313 |
| TPR-D1-H1-C160@13 | 133.37898837174529 |
| TPR-D1-H1-C160@23 | 133.05934658817205 |
| TPR-D1-H8-C160@13 | 121.8409520461135 |
| TPR-D1-H8-C160@23 | 121.86478063534273 |
| CURRENT_MEDIAN@fixed | 157.95404121750835 |
| EWMA_0P3S_H8@fixed | 159.2787954311209 |
| RISK17@7 | 146.03945113875261 |
| SIMPLEX17@7 | 144.02118652894222 |

La comparación principal conserva H8 registrado. Los controles FREE, REVERSED, LATENT_ZERO y el H16 exploratorio no se promovieron.

Tres seeds de cabeza: H8 media 121.78402164633765, std muestral 0.11983624766030283; H1 media 133.08998470088781. Delta medio de pérdidas emparejadas -11.305963054550181. CI95 jerárquico condicionado a esas seeds: [-16.21443152230172, -6.590477869051094]. No es un ensemble de TTC ni replicación de todos los expertos.

EWMA fija: mediana de fases expertas por observación, pesos exp(-lag/0,3) sobre el sufijo H8 válido, normalización y emisión registrada FP32 con conversión TTC float64. No se promediaron TTC firmados ni se barrió alpha. Cada fold conserva binding de queries, historias, ACK y productor. Sus contrastes post hoc están en [ARITHMETIC_VERIFICATION.json](supplement/ARITHMETIC_VERIFICATION.json):

```json
{
  "CURRENT_MEDIAN@fixed": {
    "score": 157.9540412175088,
    "point_delta": -1.3247542136121524,
    "hierarchical_ci95": [
      -8.691150909599731,
      4.99643590612607
    ],
    "hierarchical_fraction_negative": 0.6451416015625,
    "sequence_only": {
      "point_delta": -1.3247542136125472,
      "ci95": [
        -7.060327891013117,
        3.855165213279138
      ],
      "fraction_negative": 0.6704385735262444,
      "sequence_wins": 5,
      "sequence_count_vectors": 24310,
      "omit_one_score_deltas": [
        -1.7705070365916065,
        0.7966900389535851,
        -2.176801840293706,
        -2.9358572761542625,
        -1.8631635465746257,
        -1.4125489043319348,
        -1.472668390771517,
        -0.1262596386889996,
        -0.9616713280598574
      ],
      "signflip_p_two_sided_descriptive": 0.6875,
      "retraining_performed": false,
      "confirmatory": false
    },
    "sequence_scores": [
      180.77879838867398,
      156.35893736260698,
      257.99840718104554,
      239.53254756227986,
      103.50501381850736,
      167.0862984277196,
      87.73940544938239,
      112.61451296149765,
      115.97244980586167
    ]
  },
  "EWMA_0P3S_H8@fixed": {
    "score": 159.27879543112095,
    "point_delta": 0.0,
    "hierarchical_ci95": [
      0.0,
      0.0
    ],
    "hierarchical_fraction_negative": 0.0,
    "sequence_only": {
      "point_delta": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "fraction_negative": 0.0,
      "sequence_wins": 0,
      "sequence_count_vectors": 24310,
      "omit_one_score_deltas": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "signflip_p_two_sided_descriptive": 1.0,
      "retraining_performed": false,
      "confirmatory": false
    },
    "sequence_scores": [
      178.53753001845405,
      174.65524559674859,
      252.5067803812088,
      227.96847727555868,
      100.52249336842328,
      167.70869511557706,
      87.88084624572318,
      123.52722377449858,
      120.20186710389574
    ]
  },
  "TPR-D1-H1-C160@7": {
    "score": 132.831619142746,
    "point_delta": -26.447176288374948,
    "hierarchical_ci95": [
      -42.31601604644858,
      -11.107782368680793
    ],
    "hierarchical_fraction_negative": 0.999755859375,
    "sequence_only": {
      "point_delta": -26.44717628837483,
      "ci95": [
        -40.451099062395656,
        -12.754343790197463
      ],
      "fraction_negative": 0.9999986784385068,
      "sequence_wins": 7,
      "sequence_count_vectors": 24310,
      "omit_one_score_deltas": [
        -27.93984311280365,
        -25.768089763112936,
        -23.198269763578217,
        -22.708076152625583,
        -29.904862542258492,
        -23.382985248019835,
        -29.91346974492182,
        -27.261470360274124,
        -27.94751990777882
      ],
      "signflip_p_two_sided_descriptive": 0.015625,
      "retraining_performed": false,
      "confirmatory": false
    },
    "sequence_scores": [
      164.03168832550978,
      142.7753771062786,
      200.06835189446107,
      171.60849990118987,
      101.73680711111774,
      116.74799050436225,
      89.16401760972425,
      103.59440006131808,
      105.75743977075282
    ]
  },
  "TPR-D1-H8-C160@7": {
    "score": 121.64633225755662,
    "point_delta": -37.63246317356433,
    "hierarchical_ci95": [
      -54.579884794081124,
      -20.720944376735826
    ],
    "hierarchical_fraction_negative": 1.0,
    "sequence_only": {
      "point_delta": -37.632463173564425,
      "ci95": [
        -52.157047070493896,
        -22.97628790041567
      ],
      "fraction_negative": 1.0,
      "sequence_wins": 9,
      "sequence_count_vectors": 24310,
      "omit_one_score_deltas": [
        -39.109930102386805,
        -35.93080151459233,
        -34.06678751803308,
        -33.968450126607486,
        -41.23063904932213,
        -35.32563124480382,
        -41.44018914588802,
        -37.36834716741066,
        -40.25139269303548
      ],
      "signflip_p_two_sided_descriptive": 0.00390625,
      "retraining_performed": false,
      "confirmatory": false
    },
    "sequence_scores": [
      152.72480227546868,
      123.40948915140741,
      186.34891196339362,
      161.02390972633876,
      91.67543720092051,
      111.62157651192783,
      80.71019085074754,
      83.78183255170407,
      103.52084008609975
    ]
  }
}
```

## Factorial, interacciones y controles

| Contraste | Delta | CI95 jerárquico | CI95 secuencias |
|---|---:|---|---|
| primary_vs_RISK17@7 | -24.393118881196116 | [-38.1499414115037, -11.676666422127937] | [-36.920073482821124, -13.214495933044425] |
| primary_vs_TPR-D1-H1-C160@7 | -11.185286885189587 | [-16.137731972006225, -6.211149804697258] | [-14.818696781126025, -7.65023779770098] |
| primary_vs_REPEAT_CURRENT-D1-H8-C160@7 | -9.5648647245044973 | [-13.909119506035026, -5.350390923815288] | [-12.239913180458599, -6.899100268709059] |
| primary_vs_PAST_REVERSED-D1-H8-C160@7 | 0.42146817038036488 | [-1.7488222162877476, 2.539189367001053] | [-1.1365085974698796, 1.8996217966495328] |
| primary_vs_FREE-D1-H8-C160@7 | 0.27165650930245988 | [-2.95927025149032, 3.0419406909435738] | [-1.388621010351315, 1.91324879173213] |
| primary_vs_SELECTOR-D1-H8-C160@7 | -21.857397982393721 | [-34.38264792646343, -10.17360224163559] | [-33.064200198573694, -11.695550047033567] |
| primary_vs_CURRENT_MEDIAN_FP32@7 | -36.297135983644452 | [-54.45823179068148, -19.095967548370922] | [-53.16146098532446, -21.008254244091162] |
| TPR-D1-H4-C160_vs_primary | 5.5330298348297724 | [2.7831777575659453, 8.569824759728784] | [3.595917820928981, 7.835248311322157] |
| TPR-D1-H16-C160_vs_primary | -2.7805933473086477 | [-5.8793891043486735, 0.3014086898835507] | [-5.040727526157454, -0.29771444863622776] |
| TRANSFORMER-D1-H8-C128_vs_primary | 5.1683523937001548 | [-0.21344182086248573, 10.550543619784065] | [0.45046696024023763, 9.688464114541306] |
| LATENT-D1-H8-C160_vs_primary | 62.425967296880607 | [31.298841903806654, 98.6359525214974] | [32.65152975532086, 96.39926707448632] |
| LATENT_ZERO-D1-H8-C160_vs_primary | -0.73591647932518989 | [-2.367202281179804, 1.1548250754395157] | [-1.9178326825555212, 0.46410899876068534] |
| latent_H8_vs_H1 | 13.014966370749638 | [-0.8158395140901404, 29.545308713904237] | [-0.35045606233651305, 28.680317792950902] |
| latent_H8_vs_ZERO | 63.161883776205791 | [31.54109599911374, 99.34986779420662] | [32.95775528638231, 97.26710800233388] |
| diverse_matched_vs_dense | -9.4975351757743312 | [-16.436119574554713, -3.605480391448637] | [-12.684608985200889, -6.16512576219315] |
| primary_seed13_vs_RISK17@7 | -24.198499092639096 | [-38.35308871081571, -11.188737840961277] | [-37.153516857809414, -12.83521002653521] |
| primary_seed13_vs_TPR-D1-H1-C160@13 | -11.538036325631778 | [-16.43558875920174, -6.811802353541082] | [-14.874259869351667, -8.263452304982431] |
| primary_seed23_vs_RISK17@7 | -24.174670503409875 | [-38.79781288270447, -10.585306335411024] | [-37.79017120844973, -12.2643879376417] |
| primary_seed23_vs_TPR-D1-H1-C160@23 | -11.194565952829326 | [-16.317121710078833, -6.230112043606873] | [-15.169194653153738, -7.52180993412027] |
| three_seed_mean_loss_vs_risk | -24.255429492415033 | [-38.38412437974855, -11.131230448112634] | [-37.23055846515698, -12.800131651554771] |
| three_seed_mean_loss_vs_h1 | -11.305963054550222 | [-16.214431522301762, -6.5904778690511] | [-14.870892362624176, -7.916793662483776] |
| factor_D | -8.7572940754301101 | [-14.057197184910555, -3.859700748081862] | [-13.136277281887898, -4.934072034828902] |
| factor_H | -6.3223696743562741 | [-11.1042572726389, -1.7944071844909637] | [-9.631172326705748, -3.126758937118083] |
| factor_C | -0.38740196296835649 | [-2.192449666296175, 1.579648213882593] | [-1.5693462379272698, 0.9295522812022566] |
| factor_DxH | -7.6819720180130888 | [-11.845245161543588, -3.519804414588365] | [-10.118976951959846, -5.028557005936275] |
| factor_DxC | -4.843677131489855 | [-7.988475942963297, -1.8545734095459125] | [-6.977700411819468, -2.7988609925233865] |
| factor_HxC | -1.4547915785390926 | [-4.1080748106953155, 1.4876967567894237] | [-3.1605751635215404, 0.4576613558982088] |
| factor_DxHxC | -1.1781416502288806 | [-6.2063181930148765, 3.9852447507722837] | [-4.006703541552702, 1.8086510553420634] |

Las tablas por query, secuencia, fold y estratos, incluidos los vacíos, están incluidas junto con la receta histórica y todos los draws jerárquicos. El intervalo por secuencia no crea adquisiciones independientes nuevas.

## Conclusiones separadas

- Ejecución: T6 y transporte local verificados; cero updates en esta sesión.
- Validez: fuentes, identidades de query, normalizadores y números sellados.
- Desarrollo: H8 mejora sustancialmente la estimación en OLD_DEV reutilizado.
- Replicación: tres seeds de cabezas, mismos expertos congelados.
- Contexto: el pasado aporta información frente a H1 y REPEAT_CURRENT.
- Mecanismo: la cronología no está demostrada; REVERSED fue entrenado con inversión sistemática y puede reaprenderla.
- Seguridad/disponibilidad: no hay evidencia de AEB más rápida, tracking online persistente o incertidumbre calibrada. La GRU reinicia estado por query y el ROI depende de la query actual.
- Confirmación: Stage76, public validation, private test, EvTTC test y CodaBench permanecen cerrados.

## LATENT, regeneración y siguiente decisión

[LATENT_DIAGNOSTIC_NOTE.md](LATENT_DIAGNOSTIC_NOTE.md) separa distribuciones, productores, similitud de observaciones compartidas y errores TTC; no corrige los resultados LATENT. Contiene todas las secuencias, incluida mHGFBekt7X.

El bundle contiene los 72 pesos compactos, normalizadores, entradas normalizadas OLD_DEV para sus cabezas, predicciones y 180.000 puntos de curvas TRAIN. Los 36 checkpoints de expertos no se incluyen: se registra ruta, tamaño, hash y rol en [EXPERT_CHECKPOINT_INVENTORY.json](supplement/EXPERT_CHECKPOINT_INVENTORY.json). El alcance es reproducción de agregados y salidas de cabezas desde contextos cacheados incluidos; no reconstrucción raw-autónoma ni inferencia nueva de expertos.

Regeneración desde extracción independiente: VERIFICADA. La verificación incluye hashes internos/CRC y el script `operational/regenerate.py`. El recibo final externo vincula el SHA-256 del ZIP entregado.

[NEXT_EXPERIMENT_PROPOSAL.md](NEXT_EXPERIMENT_PROPOSAL.md) propone una única replicación acotada H16 con seis fits/15.000 updates futuros. No está ejecutada ni autorizada aquí. La literatura recuperada prepara hipótesis posteriores, sin cambiar ni retrasar la campaña.
