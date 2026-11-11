# M0 — VGGT : VRAM et temps selon le nombre d'images

Mesures brutes : [`vggt.json`](vggt.json), produites par `tools/bench_vggt.py`.
Conditions : VGGT-1B, agrégateur en bf16 (poids convertis), têtes en fp32 (elles tournent hors autocast en amont), torch 2.7.1+cu128, Windows 11. Temps = une passe avant, sans chargement ni prétraitement. Pic = `torch.cuda.max_memory_allocated`.

Budget retenu : VRAM totale − 0,5 Gio (bureau, pilote). Au-delà, Windows déborde en mémoire partagée : rien ne plante, mais tout devient très lent (fp32 d'origine sur 4 images : 7,5 Go de pic, 24 s au lieu de 1,2 s).

## RTX 3060 Laptop 6 Go (profil `degraded`, budget 5 632 Mo)

| N images | kitchen 518×350 : temps | pic | DTU scan1 518×392 : temps | pic |
|---:|---:|---:|---:|---:|
| 4 | 1,2 s | 4 010 Mo | 1,4 s | 4 117 Mo |
| 8 | 4,3 s | 4 936 Mo | 5,5 s | 5 164 Mo |
| 12 | 7,0 s | 5 070 Mo | 9,8 s | 5 309 Mo |
| 16 | 10,0 s | 5 219 Mo | 13,1 s | 5 477 Mo |
| 20 | 14,7 s | 5 351 Mo | 21,2 s | 5 622 Mo |
| 24 | — | — | 30,2 s | 5 796 Mo ✗ |
| 25 | 26,4 s | 5 535 Mo | — | — |

**N₆ = 20** pour des photos 4:3 (format smartphone, identique à DTU après redimensionnement à 518 px).

Observations :
- ~3,2 Go de poids en VRAM (agrégateur bf16 1,8 Go + têtes fp32 1,4 Go). Le saut entre 4 et 8 images vient de la tête DPT, qui traite les vues par paquets de 8.
- Au-delà de 8 images, ~35 Mo par image : la mémoire croît lentement, le temps croît en ~N² (attention globale).
- Pas de FlashAttention dans torch pour Windows : SDPA utilise le noyau « memory-efficient ». Linux devrait être plus rapide.
- Piste si N₆ = 20 ne suffit pas : passer les têtes en bf16 (~0,7 Go gagnés), à valider en précision. Sinon traitement par lots + ajustement global (plan R5).

## RTX 4060 8 Go (profil `full`)

À mesurer sur cette machine :

```bash
uv run --project worker python tools/bench_vggt.py dtu_scan1 "datasets/dtu/SampleSet/MVS Data/Rectified/scan1/rect_*_3_r5000.png" 4 8 16 24 32 40 49
```

## Précision bf16 vs fp32 (4 images kitchen)

Écart moyen entre notre chargement bf16 et la recette amont (poids fp32 + autocast bf16) : poses 1,6·10⁻⁴ (encodage ~0,4), profondeur 0,3 % relatif. Négligeable pour des poses initiales, affinées ensuite par l'ajustement de faisceaux.
