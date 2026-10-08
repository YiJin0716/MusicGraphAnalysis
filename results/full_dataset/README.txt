Full MTG-Jamendo E1-E3 data handoff (2026-10-08)

This branch adds data and provenance only. Existing 2000-track results and E4-E6 code are unchanged.
moodtheme: 18486 nodes, first connected k=6. full: 55607 nodes, first connected k=5.
All k=2..64 graphs, raw/PCA features, neighbors, weights and component memberships are provided.
components/*.npz uses labels and sizes; old code expects component_ids.
nodes.jsonl node_index aligns ALL arrays and graph rows. Largest component has labels==0.
Each experiment independently fitted preprocessing/PCA/sigma. Labels were not used for features or graphs.
Two full-set tracks lack official features: track_1057640, track_1118596; see exclusions.json.
No new E4-E6 outputs are claimed. Teammates must adjust paths, fixed 2000-node assertions, k selections,
component-key access, and E5/E6 cross-result alignment before running the existing scripts.

Release: https://github.com/YiJin0716/MusicGraphAnalysis/releases/tag/full-dataset-e1-e3-2026-10-08
manifest.json lists repository-relative destinations, file checksums and ordered Release parts.
For each group download its parts in order. Verify each part SHA-256 and byte size.
Concatenate raw groups to destination .npy; concatenate tar groups to a temporary .tar and safely unpack
repository-relative files. Verify final file SHA-256 values against manifest.json.
All part files are <=1 GiB. git clone alone DOES NOT retrieve these datasets.
Handoff groups suffice for E4-E6; dense matrices, prepared-source and official-archives preserve full data.

code/full_dataset/snapshot contains UNCHANGED generation/validation source for provenance, not adapted CLI.
To reproduce original execution, restore the original SOURCE/code, data, source_metadata, archives,
experiments layout described in ORIGINAL_RUNTIME_README.txt. check_reference expects the pilot input,
already present in the existing repository at source_data/tracks.jsonl.gz (copy locally when reproducing).
Original artifact_manifest.json and config/validation hashes describe the original run.
Report navigation links were adjusted for this branch; original reports are preserved under provenance/.
Use manifest.json as the checksum authority for distributed files. No numeric data were altered.
Official licenses/citation are in source_data/full_dataset/metadata. This collection contains no audio.
