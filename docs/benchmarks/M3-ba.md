# M3 — Ajustement de faisceaux (étape `ba`)

Mesures brutes : [`poses.json`](poses.json) (entrées du commit `9c83678`), produites par `python -m tools.eval.poses --ba`.
Seuils de rejet et départ en focale revus pour le synthétique : voir [M3-ba-synth.md](M3-ba-synth.md).
Conditions identiques à M1 : 20 vues réparties sur la série, RTX 3060 Laptop (6 Go), Windows 11.
Méthode : suivis VGGSfM de points ALIKED + SuperPoint dans des images 1024 × 1024 cadrées comme VGGT, ramenés
en pleine résolution ; départ poses et profondeurs VGGT ; caméra `SIMPLE_RADIAL` partagée ; pycolmap 3.11
(perte de Cauchy), trois ajustements avec filtrage à 4 puis 2 px (pixels de l'image de suivi).

| Dataset | Rotation médiane VGGT seul (M1) | Après `ba` | Reprojection médiane |
|---|---:|---:|---:|
| DTU (scans 1, 6) | 0,35° | **0,17°** | 0,51 px |
| MobileBrick (18 séquences) | 1,07° | **0,37°** | 0,58 px (pire 0,85) |

Critère du plan (reprojection médiane < 1 px, rotation meilleure que VGGT seul) : **atteint** sur les deux datasets.

Point principal :
- Figé au centre par défaut : sur smartphone (et MobileBrick), il est à 2-5 px du centre. Libéré, il dérive
  jusqu'à 23 px et la rotation se dégrade (MobileBrick 0,37° → 0,52°, 10 séquences sur 18 moins bonnes).
- Libéré pour DTU (`refine_principal_point`) : images rectifiées, point principal décalé d'environ 23 px.
  Figé, DTU se dégradait (0,35° → 0,66°) ; libéré, il est retrouvé à 2 px près (0,17°).

Observations :
- MobileBrick : meilleur que VGGT sur 15 séquences sur 18. `castle` reste mauvais (5,2° → 4,9°) :
  l'ajustement converge près de l'erreur de départ de VGGT, avec une reprojection correcte (0,85 px).
  `beetles` (0,46 → 0,54°), `colosseum` (0,46 → 0,48°) et `london_bus` (0,63 → 0,80°) se dégradent un peu,
  mais leur pire vue s'améliore ou reste comparable.
- Focale : écart à la vérité de +3 % (VGGT) à +1,7 % sur MobileBrick, < 0,4 % sur DTU.
- Temps : ≈ 4,5 min par série de 20 vues (suivis ≈ 3 min, ajustements ≈ 1,5 min). Pic VRAM ≈ 2,8 Go,
  avec `max_points_num` = 4096 (la valeur de VGGT, 163 840, déborde des 6 Go).
- Ajustements arrêtés à 100 itérations sans convergence formelle, coût quasi stable sur la dernière passe.
