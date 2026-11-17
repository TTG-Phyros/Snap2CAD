# M3 — `ba` sur le synthétique : cause des poses fausses et contrôles de cohérence (#97)

Mesures brutes : [`poses.json`](poses.json) (entrée du commit `33ec18e`, 30 séries synthétiques, puis DTU et
MobileBrick) et [`scale.json`](scale.json) (même commit), produites par `python -m tools.eval.poses synth:<série> ... --ba`
et `python -m tools.eval.scale`. RTX 3060 Laptop (6 Go), Windows 11, 20 vues réparties (16 à 20 selon la série).
« Avant » : l'ancien `ba` (commit `2041ad5`) relancé sur **les mêmes suivis** VGGSfM (mis en cache), pour comparer à
suivis égaux.

## Cause

Le départ de `ba` mélangeait deux géométries incompatibles :

- poses et points 3D de VGGT, cohérents avec la focale VGGT (surestimée de 2 à 15 % sur le synthétique) ;
- focale EXIF (exacte sur le synthétique).

Avec ce mélange, l'erreur de reprojection avant ajustement était de 29 à 214 px (médiane par série). Le rejet initial
à 8 px du tracker (≈ 31 px en pleine résolution) jetait alors 45 à 98 % des observations (médiane 89 %). Sur les
séries ratées, il restait quelques dizaines à quelques centaines de points (19 au minimum), vus par ≈ 2 vues chacun.
L'ajustement partait alors à la dérive (focale de 0,60 à 1,47 fois la vérité), tout en gardant une reprojection
< 1 px sur les quelques points restants.

Même avec la focale VGGT, le rejet à 8 px reste trop strict ici : 1° d'erreur de pose VGGT ≈ 13 px dans l'image du
tracker (focale ≈ 750 px) et VGGT est à 1,2° en médiane sur ce jeu (contre 0,35° sur DTU).

Ni le motif de la table, ni la petite taille de l'objet ne sont en cause : les séries ratées couvrent les trois tables
et tous les objets.

## Correction (`worker/snap2cad/steps/ba.py`)

- Rejet initial et premier ajustement avec la **focale VGGT**, libre. La focale EXIF, si toutes les caméras en ont une,
  est posée et **figée** à partir du deuxième ajustement. Libre jusqu'au bout, elle restait 1 à 2,4 % trop longue
  (variante mesurée sur 14 séries, jamais meilleure).
- Rejet initial à **64 px** du tracker (au lieu de 8), puis filtrage à **4, 2 et 1 px** (au lieu de 4 et 2). Essais sur
  `Hammer__base` : 8 px → 28 mm, 16 → 6,1, 32 → 2,8, 64 → 2,0, 128 → 2,0 mm (reprojection 1,37 px), sans rejet → 2,6 mm.
  Le filtrage final à 1 px ramène la reprojection sous 1 px sans changer les poses.
- DTU et MobileBrick n'ont pas de focale EXIF : le départ en focale VGGT ne change rien pour eux, seuls les seuils
  changent (voir plus bas).

## Résultats sur les 30 séries synthétiques

| Erreur médiane des centres de caméra | Avant | Après |
|---|---:|---:|
| < 2 mm | 6 | **21** |
| 2 à 10 mm | 12 | 6 |
| 10 à 50 mm | 6 | 3 |
| > 50 mm | 6 | **0** |

| | VGGT seul | Avant | Après |
|---|---:|---:|---:|
| Rotation médiane (médiane des séries) | 1,15° | 1,58° | **0,29°** |
| Focale / vérité (min – max) | 1,02 – 1,15 | 0,60 – 1,47 | **1,000** (EXIF) |
| Points 3D (médiane) | — | 2 110 | 10 149 |
| Reprojection médiane (médiane des séries) | — | 0,51 px | 0,65 px |

Critère M3 (reprojection médiane < 1 px) : atteint sur le jeu (0,65 px), mais 3 séries sur 30 le dépassent
(`Logitech__base` 1,15, `Razer__noir` 1,10, `Razer__base` 1,23 px). Toutes trois ont de bonnes poses (≈ 1 mm) et de longs suivis
(4 à 5 vues par point) : le seuil de 1 px en pleine résolution vaut 0,25 px dans l'image du tracker (1024 px).
Temps de l'ajustement : 38 s en médiane (suivis à part, ≈ 4 min par série, inchangés).

Par série (centres en mm, rotation en degrés ; écart à VGGT : P90 sur les vues) :

| Série | Centres | Rotation | Rotation VGGT | Reproj. (px) | Points | Écart à VGGT | Contrôles |
|---|---:|---:|---:|---:|---:|---:|---|
| `Hammer__base` | 27,9 → **1,9** | 5,57 → **0,29** | 1,56 | 0,57 | 11052 | 2,0 | — |
| `Hammer__flip` | 1,2 → **0,6** | 0,12 → **0,09** | 0,98 | 0,72 | 17042 | 1,6 | — |
| `Hammer__noir` | 46,5 → **2,0** | 5,78 → **0,49** | 1,49 | 0,60 | 9860 | 2,2 | — |
| `Mug__base` | 0,9 → **0,9** | 0,25 → **0,18** | 0,87 | 0,52 | 19020 | 1,3 | — |
| `Mug__noir` | 9,6 → **3,1** | 1,52 → **0,47** | 1,33 | 0,89 | 5489 | 2,7 | weak_views |
| `Mug__noref` | 17,9 → **2,1** | 3,11 → **0,48** | 1,41 | 0,74 | 6215 | 2,7 | — |
| `Ecoforms__base` | 105,6 → **4,0** | 10,05 → **0,32** | 0,96 | 0,51 | 6189 | 1,5 | — |
| `Ecoforms__flip` | 23,8 → **0,4** | 3,53 → **0,06** | 1,04 | 0,63 | 4519 | 2,0 | — |
| `Ecoforms__noir` | 53,6 → **33,1** | 9,09 → **4,53** | 0,81 | 0,54 | 6907 | 9,5 | weak_views, pose_drift |
| `Logitech__base` | 1,6 → **1,1** | 0,26 → **0,20** | 0,91 | 1,15 | 15021 | 0,9 | — |
| `Logitech__noir` | 9,7 → **4,8** | 1,02 → **0,61** | 0,85 | 0,67 | 4960 | 2,3 | — |
| `Logitech__noref` | 51,0 → **3,0** | 7,02 → **0,55** | 1,38 | 0,55 | 10794 | 2,4 | — |
| `Caddy__base` | 1,7 → **0,9** | 0,23 → **0,18** | 0,89 | 0,86 | 20462 | 1,4 | — |
| `Caddy__incomplete` | 5,2 → **17,0** | 1,38 → **0,46** | 1,05 | 0,49 | 8357 | 1,5 | — |
| `Caddy__noir` | 9,4 → **8,6** | 1,65 → **2,14** | 0,62 | 0,52 | 10438 | 3,6 | pose_drift |
| `OXO__base` | 9,6 → **1,4** | 1,34 → **0,24** | 1,44 | 0,69 | 8029 | 2,0 | — |
| `OXO__noir` | 9,1 → **1,2** | 1,97 → **0,31** | 1,60 | 0,73 | 5576 | 1,7 | — |
| `OXO__noref` | 60,5 → **0,5** | 8,71 → **0,09** | 1,15 | 0,65 | 15859 | 2,6 | — |
| `Pony__base` | 1,4 → **0,3** | 0,25 → **0,09** | 0,96 | 0,78 | 18829 | 1,7 | — |
| `Pony__incomplete` | 7,6 → **0,7** | 1,08 → **0,16** | 2,18 | 0,72 | 12351 | 2,5 | — |
| `Pony__noir` | 4,4 → **1,1** | 0,91 → **0,18** | 1,46 | 0,82 | 12321 | 1,7 | — |
| `Razer__base` | 2,6 → **1,2** | 0,59 → **0,23** | 1,59 | 1,23 | 6432 | 1,8 | — |
| `Razer__flip` | 1,1 → **0,5** | 0,19 → **0,10** | 1,40 | 0,72 | 20325 | 1,8 | — |
| `Razer__noir` | 2,8 → **1,1** | 0,62 → **0,29** | 1,36 | 1,10 | 14917 | 2,1 | — |
| `Shurtape__base` | 6,6 → **1,0** | 1,72 → **0,23** | 1,15 | 0,65 | 5387 | 1,7 | — |
| `Shurtape__incomplete` | 48,2 → **1,1** | 17,75 → **0,39** | 1,40 | 0,60 | 8274 | 2,0 | — |
| `Shurtape__noir` | 54,5 → **13,1** | 14,21 → **4,45** | 1,92 | 0,63 | 4082 | 15,6 | weak_views, pose_drift, focal_mismatch |
| `Teapot__base` | 218,4 → **1,6** | 24,38 → **0,43** | 1,15 | 0,53 | 10461 | 1,9 | — |
| `Teapot__incomplete` | 29,4 → **1,4** | 4,31 → **0,28** | 1,09 | 0,54 | 8235 | 1,3 | — |
| `Teapot__noir` | 5,4 → **1,4** | 1,32 → **0,31** | 0,93 | 0,56 | 10471 | 1,5 | — |

Séries encore fausses :

- `Ecoforms__noir`, `Shurtape__noir` : quelques vues finissent avec 0 à 20 observations, dérivent (jusqu'à 30°) et
  faussent toute la série (≈ 4,5° partout). Les seuils n'y changent rien (essais 32 / 64 / 128 px, filtrages [8], [4, 2]).
- `Caddy__noir` : biais uniforme de 2,1° sur toutes les vues.
- `Caddy__incomplete` (**régression**, 5,2 → 17 mm) : deux groupes de vues de part et d'autre du secteur sans photo,
  justes chacun (0,4° et 1,7°) mais mal raccordés.

## Contrôles de cohérence des poses (`ba_report.json` → `checks`, avertissements)

La reprojection ne voit pas ces échecs (0,49 à 0,63 px sur les quatre). Deux contrôles s'ajoutent à `weak_views` :

- **`pose_drift`** : écart de rotation, vue par vue, entre les poses ajustées et celles de VGGT, au changement de
  repère près (`rotation_gap`). VGGT est juste à ≈ 1° même quand `ba` dérive. Signalé si le P90 sur les vues dépasse
  3° (`max_vggt_gap_deg`). Séries justes (centres ≤ 4,8 mm) : P90 ≤ 2,7°. Détecte `Caddy__noir` (3,6°),
  `Ecoforms__noir` (9,5°) et `Shurtape__noir` (15,6°). **Ne détecte pas `Caddy__incomplete`** (1,5°) : ses rotations
  sont justes, c'est la position relative des deux groupes qui est fausse. Marge étroite (2,7° contre 3,6°), à
  réétalonner sur photos réelles.
- **`focal_mismatch`** : focale libre après le premier ajustement comparée à l'EXIF, signalée au-delà de 5 %
  (`max_focal_gap`). Séries justes : ≤ 4,7 % (`Razer__noir`). Détecte `Shurtape__noir` (8,7 %). Sert surtout pour les
  photos réelles (recadrage, zoom numérique).
- Nouveaux champs : `vggt_gap_deg` (global et par image), `focal_vggt`, `focal_exif`, `focal_free` par caméra.

## Effet sur l'échelle (étape `scale` complète, poses `ba`)

| | Avant ([M4-sigma](M4-sigma.md)) | Après |
|---|---:|---:|
| Séries où l'étape aboutit | 5 / 30 | **11 / 30** |
| Plans de table refusés faute de points | 7 | **0** |
| Refus « vues incohérentes » (rms médian > 1,5 px) | 25 | 19 |
| Normale de la table, erreur médiane | — | 0,38° (pièce), 0,31° (carte) |

- Le garde-fou de 1,5 px de `scale` refuse encore des séries aux poses justes à ≈ 1 mm (`Caddy__base` 2,15 px,
  `Logitech__base` 1,57 px, `Hammer__base` 3,27 px) : à examiner pour la décision M4 (#51), suivi dans #101.
- Parmi les 11 séries qui aboutissent, `Ecoforms__noir` (+5,29 %) et `Caddy__noir` (+0,94 %) passent les contrôles de
  `scale`, mais `ba` les signale (`pose_drift`). Sur les 9 autres : erreur d'échelle de −0,85 % à +0,30 % (carte : grand côté), médiane |0,18 %|.

## DTU et MobileBrick (non-régression)

Sans focale EXIF, seuls les seuils changent (rejet à 64 px, filtrage final à 1 px). Comparaison avec
[M3-ba.md](M3-ba.md) (entrées `9c83678` de `poses.json`, suivis recalculés à l'époque) :

| Dataset | Rotation médiane, M3 | Après | Reprojection médiane (pire) | Points 3D |
|---|---:|---:|---:|---:|
| DTU (scans 1, 6) | 0,168° | **0,161°** | 0,46 px (0,47) | +2 à +3 % |
| MobileBrick (18 séquences) | 0,367° | **0,356°** | 0,60 px (0,72) | +1 à +45 % |

- MobileBrick : meilleur ou égal sur 10 séquences sur 18 ; dégradations ≤ 0,06° (`jeep` 0,37 → 0,43°), meilleur
  gain `space_shuttle` (0,35 → 0,26°). `castle` reste mauvais (4,87°), comme VGGT (5,23°).
- Contrôles : aucun avertissement sur les 20 scènes. Écart à VGGT (P90) ≤ 0,6° sur DTU, ≤ 1,9° sur MobileBrick,
  sous le seuil de 3°. `castle` n'est pas signalé : `ba` y reste proche de VGGT, faux lui aussi.
- Temps de l'ajustement seul (suivis en cache) : 2,4 min en médiane, contre ≈ 1,5 min dans M3 (plus d'observations
  gardées au départ, une passe de plus). Pire : `dtu_scan6`, 4,5 min.

## Suites

- `ba` : vues faiblement reliées (séries `noir`, secteur sans photo) → #100.
- `scale` : garde-fou de 1,5 px trop strict avec les poses `ba` actuelles → #101, lié à #51.
