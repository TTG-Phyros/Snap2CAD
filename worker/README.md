# Worker Python `snap2cad`

## Mise en route (dev)

```bash
git submodule update --init          # third_party/ aux commits figés
cd worker && uv sync                 # Python 3.11, torch 2.7 cu128
```

Les poids sont lus dans le magasin de modèles (`%LOCALAPPDATA%/Snap2CAD/models` par défaut). En dev, pointer sur `<dépôt>/models` :

```bash
export SNAP2CAD_MODELS="$(git rev-parse --show-toplevel)/models"
```

## Commandes

```bash
uv run --project worker python tools/check_env.py      # contrôle de l'environnement
uv run --project worker pytest worker                  # tests processeur
uv run --project worker pytest worker -m gpu           # tests GPU (3060, 4060)
uv run --project worker python tools/bench_vggt.py ... # mesures VRAM/temps (docs/benchmarks)
uv run --project worker python -m tools.eval.poses     # erreur de rotation sur DTU / MobileBrick / synth:<série> (docs/benchmarks/poses.json)
uv run --project worker python -m tools.eval.segment   # IoU des masques sur le synthétique, repli 3D et BiRefNet compris (docs/benchmarks/segment.json)
uv run --project worker python -m tools.eval.birefnet  # BiRefNet seul sur le synthétique (docs/benchmarks/birefnet.json)
```

Au premier chargement, VGGT est converti en bf16 dans `<magasin>/vggt-1b-bf16/` (≈ 2,5 Go, une seule fois).

## Jeu synthétique (M1b, projet uv séparé : Blender 4.2 en module `bpy`)

```bash
uv run --project tools/synth python tools/synth/fetch_gso.py               # 10 objets GSO → datasets/gso/
uv run --project tools/synth python -m tools.synth.generate --skip-existing  # 10 objets × 3 variantes → datasets/synth/
uv run --project tools/synth python -m tools.synth.generate Pony_C_Clamp_1440__flip  # une série
```
