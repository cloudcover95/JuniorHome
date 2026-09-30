# JuniorDeck modular port

Taken from the audit, not the theater.

Phase 1 mech: config/deck_port.toml pin 2.54 mm 2x8, mount 24 mm.
Phase 2 analog: CV \pm5 V, audio 2 Vpp, Z 10 k\u03a9, 48 kHz. analog_ok before digitizer.
Phase 3 trit-energy: existing pack_wave + jsonl ~/.juniorhome/gaia_mesh/deck.jsonl.
Phase 4 gate: scripts/osai_gate_prod.py Flagstaff 6-vote AND.

Blocked: Rust FFI, 2048 SVD, parquet Home, 0.0.0.0, CUDA graphs,
ue5_launch, downloads, I2_S as chain address, rename absmean_pack.
ml_kem false. JuniorSOL live, JuniorSolana archive.
