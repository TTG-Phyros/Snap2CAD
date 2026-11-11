# M1 — Erreur de poses de VGGT seul (référence pour M3)

Mesures brutes : [`poses.json`](poses.json), produites par `python -m tools.eval.poses`.
Conditions : étape `poses_init` (VGGT-1B bf16), 20 vues réparties sur la série, RTX 3060 Laptop, Windows 11.
Calcul : similitude (Umeyama) entre les centres des caméras estimés et vrais, puis angle de `R_est·R_vraiᵀ` par vue.
DTU : images rectifiées, éclairage 3 (diffus), sans EXIF (focale par défaut). MobileBrick : 98 images 1920 × 1440 par séquence.

| Dataset | Rotation médiane (médiane des scènes) | Pire scène | Centre (médiane) |
|---|---:|---|---:|
| DTU (scans 1, 6) | **0,35°** | scan1 0,37° | 2,2 – 2,6 mm |
| MobileBrick (18 séquences) | **1,07°** | castle 5,23° | 2,9 – 13 mm |

Cible du plan (§9.4) : < 1° **après** ajustement de faisceaux (M3). VGGT seul l'atteint sur DTU, pas sur MobileBrick.

Observations :
- `castle` est le seul cas franchement mauvais (5,2°, 13 mm) ; les 17 autres séquences sont entre 0,46° et 2,4°.
- Focale VGGT surestimée de 0,5 à 8 % (médiane ≈ +3 %) : l'ajustement M3 doit libérer la focale.
- Temps : 20 à 33 s par scène, chargement du modèle compris.
