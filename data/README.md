# Política de datos

Los datasets crudos no se almacenan en Git.

## IDs permitidos

```text
EVTTC32_LABELLED
EAP_HF_TRAIN40
BENCHMARK10_SEALED
```

No se aceptan nuevas raíces sin una revisión explícita del protocolo.

## Contenido versionable

- manifests pequeños;
- splits legibles;
- hashes;
- fixtures sintéticos;
- documentación.

Cada entrada de manifest debe incluir fuente, ruta local, tamaño, metadata de
split y hash cuando esté disponible.

## Rutas operativas

```text
datasets/evttc
datasets/evttc_official_benchmark_sealed
E:\eAP_dataset\data\train
E:\eAP_dataset\derived
```

La raíz sellada no se inspecciona durante desarrollo. eAP derived está limitado
a 55 GiB y debe mantener al menos 50 GiB libres en E:.

## Prohibiciones

- datos crudos en Git;
- eAP test;
- extracción masiva de TAR RGB;
- voxel cache global;
- logits SAM densos;
- hidden states DINO de todas las capas;
- pseudo-TTC presentado como ground truth.
