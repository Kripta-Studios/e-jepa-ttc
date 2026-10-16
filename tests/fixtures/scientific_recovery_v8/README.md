# V8 freezer fixtures

`dino_teacher_manifest.json` preserves the historical metadata bytes referenced by the grouped A5 configuration (SHA-256 `6ee4b205d29a07d3aabf7137f2a312376711b5e5d9d7930ea36328390023eab3`). It contains metadata and checksums, not cached tensors, images, or model weights.

The V8 freezer tests generate synthetic OOF rows and checksum-only checkpoint files in a temporary repository. They exercise the real signature, hash, population, fold and tamper checks. Those files are test inputs, never research results. Production configuration files are not rewritten by the tests.

`essential_results_manifest.json` preserves the original 1,517-entry V8 archive manifest (SHA-256 `86b97a61232e6d260ee273009460981c9a1d1e75f1588a4b5c41432882ddc378`). This is metadata only, not the 317 MB results archive. Clean-clone CI always verifies manifest identity; full member verification runs only where the original external results are available.
