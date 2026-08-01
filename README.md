# FastSD

FastSD is a research codebase for speculative decoding under cloud-edge collaboration. The current repository mixes three layers in one place:

- core decoding and KV-cache logic in `src/`
- benchmark and comparison entrypoints in `benchmark/` and `comparison/`
- a newer cloud/edge service path in `cloud/` and `edge/`

This repository should be read with one important distinction in mind:

- the paper goal describes the intended FastSD system and its research contributions
- the current codebase is a partial, evolving implementation of that goal

Do not assume every paper feature is already fully implemented, and do not assume every file in the repo maps cleanly to the final paper narrative.

If you are new to the repo, read the detailed guide first:

- [docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md)
- [docs/CAMPUS_SERVER_EXTERNAL_ACCESS.md](docs/CAMPUS_SERVER_EXTERNAL_ACCESS.md)

Quick pointers:

- Install dependencies: `bash install.sh`
- Start cloud target service: `python cloud/cloud_service.py --exp_name <name>`
- Start edge runner with a preset profile: `bash scripts/run_fastsd_profile.sh <exp_name>`
- Run tests: `python -m unittest tests.test_energy_meter tests.test_fastsd_scheduler -v`

SpecEdge paper baseline:

- Pinned official implementation and paper-aligned configs: [baselines/specedge/README.md](baselines/specedge/README.md)
- Baseline control-layer tests: `python -m unittest tests.test_specedge_repro -v`

The repository currently has local uncommitted changes copied from the remote development server. Check `git status --short` before assuming a clean baseline.
