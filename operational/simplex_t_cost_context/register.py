"""Admit immutable historical H8 sources and seal the prospective 18-fit study."""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path

from operational.simplex_t_h16_replication.common import (
    HISTORICAL_FREEZE,
    inventory,
)

from .engine import (
    ARM_ORDER,
    CAMPAIGN,
    N1,
    OUT,
    ROOT,
    Resources,
    atomic_bytes,
    digest,
    historical_spec,
    ids,
    publish_json,
    record,
    sources,
    validate_pins,
)

PROSE = """# Protocolo prospectivo independiente de precisión/coste/contexto

Autoridad: ampliación explícita del usuario y paquete nocturno verificado.
T6 y TPR-D1-H8-C160 permanecen cerrados. N1 conserva su receta H16 completa.
Este protocolo autoriza únicamente N2/N3:18fits y45000updates, incluidos en
la unión24fits/60000updates científicos. No se reutiliza presupuesto antiguo.

Orden inmutable:FULL_C0,A5_ONLY_C0,C2F_ONLY_C0,A5_PAIR_C0,SET_AGE_C0,SET_NOTIME_C0;
folds0,1,2 de cada brazo, seed7. H8/D1, mismos TRAIN queries y pesos originales,
normalizadores diagonales TRAIN históricos, misma PHASE17 y ROI actual retrospectivo.
Cada cabeza nace desde inicialización por seed; ningún warm start o destilación.
CPU FP32,batch128,cuatro threads,interop2; AdamW wd1e-3,beta(.9,.999),eps1e-8,
foreachFalse; warmup100/cosine2500,lr3e-4→3e-5,clip1 yendpoint2500.
Único cambio de objetivo común:L_phase+.1L_quant;lambda_cost=0 en los seis brazos.
El driver histórico se conserva byte a byte; fábrica/loss se enlazan sólo en
el proceso nuevo mediante contexto reversible. El envelope config conserva GRU160;
el arm y su arquitectura real quedan fijados por protocolo/source identity.

FULL_C0:17features,GRU160x2,mediana actual tres expertos.
A5_ONLY_C0:6features,GRU160x2,anchor A5.
C2F_ONLY_C0:6features,GRU160x2,anchor C2F.
A5_PAIR_C0:9features,GRU160x2,punto medio fases A5/PAIR (no media TTC).
Slots excluidos se anulan DESPUÉS de la normalización; sus diagnósticos,
diferencias,anchor y supervisión auxiliar se eliminan. Índice/validez histórica
común no se filtra por experto ni por resultados. Sus dependencias online no
quedan certificadas por enmascarar cachés; se declara esa limitación.
SET_AGE_C0 ySET_NOTIME_C0:pooling exacto del paquete,289765parámetros frente
a313765GRU. Mean/max del pasado,actual distinguido,fraction valid;sin posición.
SET_AGE usa parejas feature-tiempo;SET_NOTIME anula cuatro tiempos explícitos.
Se conserva emisión TTC/quantiles canónica. Cost heads están inactivos.

Registro previo de contratos,IDs,hashes,comparadores y análisis; ningún ajuste
usando los nuevos scores. N2 congela doce endpoints antes de su comparación;
N3 congela seis. Si deadline deja familia incompleta, inventario previo a
evaluación parcial únicamente de brazos con sus tres folds2500 completos,
marcados exploratorios. Jamás interpretar un checkpoint parcial como endpoint.

Contrastes fijados:FULL_C0-H8seed7; cada reducido-FULL_C0;
SET_AGE_C0-FULL_C0;SET_NOTIME_C0-SET_AGE_C0. Pérdidas emparejadas query a query,
masas/buckets OLD_DEV completas,8192queries,nueve secuencias,bootstrap jerárquico
secuencia/track con draws T2 sellados8192 yfragmentos256,contraste por secuencias.
Semilla única N2/N3, reutilización desarrollo/múltiples contrastes exploratorios.
Reportar global,ambosCI95,fold,secuencia,bucket crucial,signo,cobertura,cap,
escapes beneficiosos/perjudiciales ycasos vacíos. No selección del mejor score.

Cribado NUEVO ingenieril sin promoción:excesoMiD vsFULL_C0 límite superiorCI95<2,
delta error signo≤.005,delta crucial≤5MiD,p95ruta relevante≤.8comparador.
Cabeza sola sólo permite conclusión de eficiencia de cabeza; sin coste completo
no se decide sustitución del sistema. Negativo/incierto se entrega sin rescate.

Un trainer pesado,writer propio,checkpoint completo100 yprogreso25. Restore incluye
optimizador,RNG,sampler,contador yLR derivado del scheduler histórico. Pruebas
sintéticas FULL_C0 ySET_AGE:20continuo frente10+10 reanudado por familia,80updates
técnicos totales,cap200 separado. Unión pérdidas/repeticiones≤6000;techo físico66200.
Ventana absoluta/deadline sellados enWINDOW_AUTHORIZATION.json. Recursos:
RAMdisponible≥2GiB,RSSautorizado≤4GiB,disco libre-reserva1GiB≥10GB;
compromiso Windows registrado yflooroperativo1GiB. Artefactos propios≤2GiB.

Latencia cabeza:selección determinista porhashIDsTRAIN,batch1,warmup25 y500medidas
si cabe;batch128throughput separado. Contexto/ROI/productores/raw sólo con slot
exclusivo autorizado yfuentes disponibles;sin ello costerescope parcial explícito.
PAIR reutilizaA5, no se cuenta comootroencoder. No estadoGRUpersistente ni ahorro
de ROI supuesto. No claimsAEB,tracking online,energía ocalibración probabilística.
No push,nuevos datos/encoders/LATENT/papers/sweeps/seeds,niStage70–76/holdouts.
"""


def main() -> None:
    """Validate TRAIN and parent identities without any optimizer or new scoring."""
    import numpy as np
    import torch

    from .model import FEATURE_NAMES, MaskedSource

    resources = Resources()
    resources.check()
    parent_path = N1 / "PROTOCOL.json"
    parent_pin = (N1 / "PROTOCOL.sha256").read_text().split()[0]
    parent = record(parent_path)
    if digest(parent_path) != parent_pin or parent["historical_freeze_sha256"] != HISTORICAL_FREEZE:
        raise ValueError("independent N1 parent protocol/freeze changed")
    validate_pins(parent, full=True)
    if digest(Path(parent["launch"]["freeze"])) != HISTORICAL_FREEZE:
        raise ValueError("historical scientific freeze changed")
    window = record(OUT / "WINDOW_AUTHORIZATION.json")
    if (
        window["maximum_scientific_saved_updates"] != 60000
        or window["maximum_scientific_fits"] != 24
    ):
        raise ValueError("union authorization differs")
    admission: dict = {}
    s = sources(parent)
    for fold in range(3):
        resources.check()
        spec = historical_spec(s, fold, 8)
        train, dev = s.source(spec, "inner_oof"), s.source(spec, "outer_dev")
        if train.identity_sha256 != parent["sources"][str(fold)]["h8_train_sha256"]:
            raise ValueError("H8 TRAIN source differs from independently admitted N1 controls")
        if train.length != 8 or dev.length != 8 or train.features.shape[1] != 17:
            raise ValueError("D1/H8 PHASE17 parent required")
        if not (
            np.array_equal(train.normalizer.mean, dev.normalizer.mean)
            and np.array_equal(train.normalizer.scale, dev.normalizer.scale)
            and train.normalizer.consumed_ids_sha256 == dev.normalizer.consumed_ids_sha256
        ):
            raise ValueError("TRAIN/OLD_DEV historical normalizers differ")
        # Gather QA touches only TRAIN features and targets; dev is identity-only.
        ids_train = torch.arange(min(128, train.population))
        expected = train.gather(ids_train)
        if any(not bool(torch.isfinite(v).all()) for i, v in enumerate(expected) if i != 2):
            raise ValueError("nonfinite verified TRAIN minibatch")
        wrapped: dict = {}
        for arm in ARM_ORDER:
            a, b = MaskedSource(train, arm), MaskedSource(dev, arm)
            gathered = a.gather(ids_train)
            if (
                a.population != train.population
                or not torch.equal(gathered[4], expected[4])
                or not torch.equal(gathered[5], expected[5])
                or not torch.equal(gathered[2], expected[2])
            ):
                raise ValueError("prospective mask changed population/targets/masses/validity")
            wrapped[arm] = dict(train_sha256=a.identity_sha256, dev_sha256=b.identity_sha256)
        import hashlib

        admission[str(fold)] = dict(
            parent_train_sha256=train.identity_sha256,
            parent_dev_sha256=dev.identity_sha256,
            train_population=train.population,
            old_dev_population=dev.population,
            normalizer_mean_sha256=hashlib.sha256(train.normalizer.mean.tobytes()).hexdigest(),
            normalizer_scale_sha256=hashlib.sha256(train.normalizer.scale.tobytes()).hexdigest(),
            normalizer_consumed_ids_sha256=train.normalizer.consumed_ids_sha256,
            wrapped=wrapped,
        )
        del train, dev, expected, gathered, a, b
        s.release()
        gc.collect()
    code = list(Path(__file__).parent.glob("*.py"))
    receipts = [
        OUT / "verification/MODEL_TESTS.xml",
        OUT / "verification/MODEL_TESTS_RESOURCE.json",
        OUT / "verification/ENGINE_RESOURCE_QA.xml",
        OUT / "verification/DELIVERY_RECONCILIATION_QA.xml",
    ]
    if any(not path.exists() for path in receipts):
        raise ValueError("focused model QA receipts required")
    comparators = [
        path for path in (OUT / "BASELINE_REGISTRY.csv", OUT / "COMPARABILITY.md") if path.exists()
    ]
    pins = list(parent["input_inventory"]) + inventory(
        code
        + receipts
        + comparators
        + [
            parent_path,
            N1 / "PROTOCOL.sha256",
            OUT / "WINDOW_AUTHORIZATION.json",
            OUT / "input_package/PROTOCOLO_COSTE_Y_CONTEXTO.md",
            OUT / "input_package/reference/efficiency_contracts.py",
        ]
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    p = dict(
        schema="prospective_cost_context_protocol_v1",
        campaign=CAMPAIGN,
        training_commit=head,
        scientific_code=record(Path(parent["launch"]["freeze"]))["code_commit"],
        historical_candidate="TPR-D1-H8-C160",
        historical_freeze_sha256=HISTORICAL_FREEZE,
        fits=ids(),
        scientific_updates=45000,
        lambda_cost=0.0,
        lambda_quant=0.1,
        new_study_technical_updates_authorized=200,
        planned_resume_test_updates=80,
        launch=parent["launch"],
        historical_controls=parent["historical_controls"],
        sources=admission,
        feature_names=list(FEATURE_NAMES),
        input_inventory=pins,
        parent_h16_protocol_sha256=parent_pin,
        bootstrap_draws=parent["bootstrap_draws"],
        window_authorization=window,
        source_pin=dict(commit=head, files=inventory(code)),
        comparisons=[
            dict(candidate=a, comparator=b)
            for a, b in (
                ("FULL_C0", "H8@7"),
                ("A5_ONLY_C0", "FULL_C0"),
                ("C2F_ONLY_C0", "FULL_C0"),
                ("A5_PAIR_C0", "FULL_C0"),
                ("SET_AGE_C0", "FULL_C0"),
                ("SET_NOTIME_C0", "SET_AGE_C0"),
            )
        ],
        profiling_design=dict(
            selection=(
                "64 TRAIN fold0 queries by SHA256 of real sample_token; profile.py authoritative"
            ),
            code=str(Path(__file__).parent / "profile.py"),
            code_sha256=digest(Path(__file__).parent / "profile.py"),
            warmups=25,
            measurements=500,
            batch=1,
            threads=4,
            cpu_fp32=True,
            throughput_batch128_reported_separately=True,
            end_to_end_only_if_sources_and_exclusive_slot_available=True,
            no_target_or_error_selection=True,
        ),
        guardrails=dict(
            upper_ci_excess_mid_strict=2.0,
            sign_error_delta_max=0.005,
            crucial_mid_delta_max=5.0,
            relevant_p95_ratio_max=0.8,
            scope_required_for_system_substitution="END_TO_END",
        ),
        comparator_registry=dict(
            paths=[str(v) for v in comparators], missing_rows_do_not_block_independent_fits=True
        ),
        protocol_text=PROSE,
        holdouts_opened=False,
        scores_of_new_fits_read=False,
        no_push=True,
    )
    publish_json(OUT / "PROTOCOL_COST_CONTEXT.json", p)
    atomic_bytes(OUT / "PROTOCOL_COST_CONTEXT.md", PROSE.encode())
    pin = digest(OUT / "PROTOCOL_COST_CONTEXT.json")
    atomic_bytes(
        OUT / "PROTOCOL_COST_CONTEXT.sha256", (pin + "  PROTOCOL_COST_CONTEXT.json\n").encode()
    )
    publish_json(
        OUT / "SOURCE_PIN_COST_CONTEXT.json",
        dict(training_commit=head, protocol_sha256=pin, files=inventory(code)),
    )
    print("COST_CONTEXT_PROTOCOL_SEALED " + pin, flush=True)


if __name__ == "__main__":
    main()
