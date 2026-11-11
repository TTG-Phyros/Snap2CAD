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
```

Au premier chargement, VGGT est converti en bf16 dans `<magasin>/vggt-1b-bf16/` (≈ 2,5 Go, une seule fois).
