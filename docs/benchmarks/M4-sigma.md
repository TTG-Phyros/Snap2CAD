# M4 — σ d'échelle et contrôles de cohérence (`scale.sigma_rel`, `scale.checks`)

Mesures brutes : [`scale.json`](scale.json) (entrée du commit `913af89`), produites par `python -m tools.eval.scale`
(30 séries synthétiques). Deux cas : caméras exactes (`fit3d.exact`, table z = 0) et étape complète avec les
poses `ba` (`fit3d.step`, vérité par similitude des centres des caméras).

## Méthode

- **σ des vues (jackknife) :** le cercle ou le rectangle 3D est réajusté sans chaque vue gardée, à partir de la
  solution. Les tailles sont paramétrées en logarithme : l'écart type jackknife est directement relatif.
- **Plancher :** le jackknife ne voit que la dispersion entre vues. Le biais du contour commun à toutes les vues
  lui échappe : avec les caméras exactes, seules 4 pièces sur 21 et 4 cartes sur 9 sont dans 1σ (médiane de
  |erreur| / σ ≈ 1,8). On ajoute donc un plancher en quadrature : σ = √(σ_vues² + σ₀²), avec σ₀ = 0,25 %
  (pièce) et 0,05 % (carte, grand côté), choisis pour avoir ≈ 68 % des erreurs dans 1σ.
- **Contrôles** (`scale.json` → `checks`, `checks_ok` ; signalés, sans échec de l'étape) :
  - hauteur de la face haute au-dessus du plan de table − épaisseur de la référence : ≤ 1 mm (0,2 mm avant #102,
    section dédiée plus bas) ;
  - angle entre la normale de la référence et celle de la table : ≤ 1° ;
  - carte : échelle du petit côté vs grand côté : ≤ 1 % (biais du petit côté jusqu'à 0,57 % avec les caméras
    exactes, [M4-fit3d.md](M4-fit3d.md)) ;
  - avertissements de `ba` (`ba_warnings`) : tout avertissement rend `checks_ok` faux (depuis #102).

## Résultats

Étalonnage de σ (caméras exactes) :

| | Séries | σ médian | \|erreur\| / σ médian | Dans 1σ | Dans 2σ |
|---|---:|---:|---:|---:|---:|
| Pièce | 21 | 0,27 % | 0,68 | 13 (62 %) | 20 (95 %) |
| Carte (grand côté) | 9 | 0,054 % | 0,73 | 6 (67 %) | 9 (100 %) |

Contrôles, caméras exactes : 29 séries sur 30 passent. Écarts maximaux : hauteur 0,04 mm (carte), 0,33 mm
(pièce, `Ecoforms__flip` seule, 5 vues, erreur d'échelle −0,87 %) ; inclinaison 0,45° hors cette série (1,4°) ;
côtés de la carte 0,57 %.

Étape complète avec l'ancien `ba` (commit `913af89`, avant #97), les 5 séries qui aboutissent :

| Série | Erreur d'échelle | σ | Hauteur − épaisseur | Inclinaison | Contrôles |
|---|---:|---:|---:|---:|---|
| `Razer__flip` (pièce 1 €) | +0,04 % | 0,32 % | −0,11 mm | 0,48° | ok |
| `Ecoforms__noir` (pièce 1 €) | +0,86 % | 0,28 % | **−0,34 mm** | 0,49° | signalé |
| `Logitech__base` (carte) | +1,22 % | 0,09 % | **−0,65 mm** | 0,18° | signalé |
| `Caddy__incomplete` (pièce 2 €) | +1,85 % | 0,40 % | **−0,42 mm** | 0,60° | signalé |
| `Teapot__noir` (pièce 2 €) | +1,97 % | 0,28 % | +0,09 mm | 0,11° | **ok (faux)** |

- La hauteur de la référence est le contrôle utile : elle sépare la bonne série (0,11 mm) de trois fausses
  (≥ 0,34 mm). L'inclinaison et l'accord des côtés de la carte ne détectent rien ici.
- `Teapot__noir` passe tous les contrôles avec +1,97 % : les poses `ba` y sont fausses de façon cohérente
  (similitude globale), invisible de l'intérieur de l'étape. Seule la correction de `ba` (ticket M3) y répond.
- Avec les poses `ba`, σ sous-estime l'erreur (médiane |erreur| / σ = 3,9 pour la pièce) : il ne couvre que
  l'erreur du contour, pas celle des poses.

## Garde-fou « vues incohérentes » relatif à la taille de la référence (#101)

Entrée `6e542c6` de [`scale.json`](scale.json) : poses `ba` corrigées (#97, [M3-ba-synth.md](M3-ba-synth.md)).
« Centres » : erreur médiane des centres des caméras `ba` après similitude (mm).

**Cause.** Le garde-fou refusait l'étape si le rms médian des vues gardées dépassait 1,5 px. Un rms en px ne
dit rien sans la taille de la référence dans l'image : rayon de la pièce 60 à 180 px, demi grand côté de la
carte 280 à 400 px. Le même écart de pose donne un rms comparable sur les deux, mais une erreur d'échelle 3 à
5 fois plus grande sur la pièce. Le seuil porte désormais sur `rms_rel` = rms médian / taille projetée médiane
(rayon ou demi grand côté) des vues gardées, avec un échec au-delà de **5 %** (`FIT3D["max_rms_rel"]`).

**Résidu attendu pour une erreur de pose.** Caméras exactes perturbées par un bruit indépendant par vue
(5 tirages, médiane ; focale 2912 px) :

| Bruit par vue | `Hammer__base` (pièce 1 €) | `Caddy__base` (pièce 2 €) | `OXO__base` (carte) |
|---|---:|---:|---:|
| aucun | 0,26 px · 0,34 % · 0,08 % | 0,24 px · 0,29 % · 0,26 % | 0,22 px · 0,06 % · 0,08 % |
| centres 0,5 mm | 2,0 px · 2,3 % · 0,15 % | 1,9 px · 2,3 % · 0,59 % | 2,4 px · 0,66 % · 0,11 % |
| centres 1 mm | 3,7 px · 4,1 % · 0,92 % | 3,6 px · 4,4 % · 0,86 % | 4,8 px · 1,33 % · 0,09 % |
| centres 2 mm | 7,5 px · 8,5 % · 1,91 % | 6,9 px · 8,4 % · 1,56 % | 8,9 px · 2,47 % · 0,07 % |
| rotations 0,1° | 2,7 px · 3,0 % · 0,57 % | 2,4 px · 3,0 % · 0,83 % | 2,0 px · 0,57 % · 0,10 % |
| rotations 0,3° | 7,3 px · 8,2 % · 1,36 % | 6,7 px · 8,3 % · 1,68 % | 6,2 px · 1,71 % · 0,07 % |
| 1 mm et 0,3° | 7,6 px · 8,9 % · 1,08 % | 7,1 px · 8,9 % · 2,02 % | 8,5 px · 2,38 % · 0,12 % |

(rms médian · `rms_rel` · |erreur d'échelle|)

- Un bruit indépendant de 1 mm et 0,3° par vue donnerait 7 à 9 px. Les poses `ba` réelles (≈ 1 mm, 0,3°) ne
  donnent que 1 à 3,5 px : leur erreur est surtout commune aux vues (proche d'une similitude), sans effet sur
  l'ajustement de la référence.
- La pièce perd ≈ 0,9 % d'échelle par mm de bruit indépendant ; la carte (4 coins, plus grande) reste ≤ 0,12 %.

**Résultats (étape complète, 30 séries).** Caméras exactes : `rms_rel` ≤ 0,70 %.

| | Avant (1,5 px) | Après (5 %) |
|---|---:|---:|
| Séries où l'étape aboutit | 11 / 30 | **28 / 30** |
| dont poses justes (centres < 2 mm) | 7 / 21 | **21 / 21** |
| `rms_rel` max, poses justes | — | 3,66 % (`Hammer__base`) |
| `rms_rel` des séries refusées | — | 13,8 % (`Ecoforms__base`, 4,0 mm), 29,3 % (`Caddy__incomplete`, 17 mm, −48 % sans garde-fou) |

Les 9 séries aux poses fausses (centres ≥ 2 mm) sont toutes refusées ou signalées : 2 refusées par le
garde-fou, 4 signalées par `ba` (`weak_views`, `pose_drift`), et les 3 restantes (`Mug__noref`,
`Logitech__noref`, `Logitech__noir`) par les contrôles de `scale`, avec des erreurs faibles (−0,18 à +0,11 %).

| Série | Référence | Centres (mm) | `rms_rel` | Erreur d'échelle | σ | Contrôles `scale` | Avertissements `ba` |
|---|---|---:|---:|---:|---:|---|---|
| `Pony__base` | pièce 2 € | 0,35 | 0,93 % | +0,20 % | 0,30 % | ok | — |
| `Ecoforms__flip` | pièce 1 € | 0,41 | 1,07 % | −0,85 % | 0,32 % | signalé | — |
| `OXO__noref` | carte | 0,48 | 0,60 % | −0,03 % | 0,10 % | ok | — |
| `Razer__flip` | pièce 1 € | 0,50 | 1,02 % | +0,09 % | 0,34 % | ok | — |
| `Hammer__flip` | pièce 1 € | 0,59 | 1,15 % | +0,27 % | 0,33 % | ok | — |
| `Pony__incomplete` | pièce 2 € | 0,66 | 1,76 % | +0,42 % | 0,32 % | signalé | — |
| `Caddy__base` | pièce 2 € | 0,89 | 2,63 % | +1,15 % | 0,51 % | signalé | — |
| `Mug__base` | carte | 0,89 | 0,46 % | −0,01 % | 0,06 % | ok | — |
| `Shurtape__base` | pièce 2 € | 0,96 | 1,44 % | +0,45 % | 0,49 % | signalé | — |
| `Razer__noir` | pièce 1 € | 1,06 | 1,15 % | −0,05 % | 0,38 % | signalé | — |
| `Shurtape__incomplete` | pièce 2 € | 1,07 | 1,27 % | +0,66 % | 0,74 % | signalé | — |
| `Logitech__base` | carte | 1,08 | 0,56 % | −0,03 % | 0,06 % | signalé | — |
| `Pony__noir` | pièce 2 € | 1,08 | 2,31 % | +0,79 % | 0,48 % | signalé | — |
| `OXO__noir` | carte | 1,16 | 0,63 % | −0,09 % | 0,08 % | signalé | — |
| `Razer__base` | pièce 1 € | 1,21 | 2,45 % | −0,10 % | 0,44 % | signalé | — |
| `Teapot__noir` | pièce 2 € | 1,35 | 1,29 % | +0,03 % | 0,29 % | ok | — |
| `Teapot__incomplete` | pièce 2 € | 1,39 | 2,81 % | −0,34 % | 0,32 % | ok | — |
| `OXO__base` | carte | 1,40 | 0,48 % | +0,05 % | 0,14 % | signalé | — |
| `Teapot__base` | pièce 2 € | 1,64 | 3,42 % | +0,85 % | 0,69 % | signalé | — |
| `Hammer__base` | pièce 1 € | 1,95 | 3,66 % | +1,71 % | 0,95 % | signalé | — |
| `Hammer__noir` | pièce 1 € | 1,96 | 1,65 % | +0,30 % | 0,46 % | signalé | — |
| `Mug__noref` | carte | 2,08 | 0,35 % | −0,18 % | 0,11 % | signalé | — |
| `Logitech__noref` | carte | 3,05 | 0,47 % | −0,14 % | 0,09 % | signalé | — |
| `Mug__noir` | carte | 3,14 | 0,61 % | −0,68 % | 0,22 % | signalé | weak_views |
| `Ecoforms__base` | pièce 1 € | 3,98 | **refusé** (13,8 %) | — | — | — | — |
| `Logitech__noir` | carte | 4,75 | 0,83 % | +0,11 % | 0,66 % | signalé | — |
| `Caddy__noir` | pièce 2 € | 8,61 | 1,52 % | +0,94 % | 0,29 % | ok | pose_drift |
| `Shurtape__noir` | pièce 2 € | 13,09 | 0,97 % | +2,25 % | 0,35 % | signalé | weak_views, pose_drift, focal_mismatch |
| `Caddy__incomplete` | pièce 2 € | 17,01 | **refusé** (29,3 %) | — | — | — | — |
| `Ecoforms__noir` | pièce 1 € | 33,15 | 1,42 % | +5,29 % | 0,33 % | ok | weak_views, pose_drift |

Sur les 21 séries aux poses justes :

| | Pièce (16) | Carte, grand côté (5) |
|---|---:|---:|
| \|erreur\| médiane | 0,38 % | 0,03 % |
| \|erreur\| P90 / max | 1,00 % / 1,71 % | 0,08 % / 0,095 % |
| Sous la cible M4 | 4 / 16 (< 0,15 %) | 4 / 5 (< 0,08 %) |
| σ médian | 0,41 % | 0,08 % |
| Dans 1σ / 2σ | 9 / 14 | 4 / 5 |
| Contrôles `scale` ok | 5 / 16 | 2 / 5 |

- Le garde-fou ne retient que l'incohérence grossière : une pièce à `rms_rel` 3,66 % sort avec +1,71 %, mais
  avec σ = 0,95 % (dans 2σ). Le jackknife suit maintenant l'incohérence des vues : σ couvre 14 pièces sur 16
  dans 2σ, contre une médiane |erreur| / σ de 3,9 avec l'ancien `ba`.
- Le contrôle de hauteur de la référence donne beaucoup de fausses alertes avec les poses `ba` (jusqu'à
  −2,6 mm sur `Razer__base`, erreur −0,10 %) : 7 séries justes sur 21 seulement passent tous les contrôles.
  Corrigé par #102 (section suivante).
- `Ecoforms__noir` (+5,29 %) et `Caddy__noir` (+0,94 %) passent les contrôles de `scale` : seul `ba` les signale.

## Contrôle de hauteur : plan de table et avertissements de `ba` (#102)

Entrée `758cc4c` de [`scale.json`](scale.json), mêmes 30 séries et mêmes poses `ba` qu'en `6e542c6`.

**Cause des fausses alertes.** Deux mesures pour séparer le plan de table du cercle ou rectangle 3D :

- Le plan RANSAC suit les points de table proches de la référence. Leur hauteur médiane au-dessus du plan, à
  moins de 1,5 rayon de la référence, est de −0,01 mm sur `Razer__base` et `Shurtape__base`, et de +0,06 mm sur
  `Hammer__noir`. RANSAC n'est donc pas en cause.
- Les points de table eux-mêmes sont trop hauts, **même avec les caméras exactes**. Retriangulés à partir des
  mêmes observations 2D, ils sortent à z = +2,0 mm (`Razer__base`) et +2,2 mm (`Shurtape__base`) au lieu de 0. À
  moins de 30 mm de la référence : +2,5 mm. Ce sont des points de la face haute de la pièce (épaisseur 2,2 à
  2,3 mm). Le masque objet les classait comme table. Sur une table peu texturée, ils pèsent assez pour relever
  le plan de l'épaisseur de la pièce : hauteur − épaisseur ≈ −2,3 mm.
- Les écarts restants, jusqu'à 1 mm, viennent des poses `ba`. Sur `Hammer__noir`, les points de table sont à
  +0,7 mm avec les caméras exactes et à +5,4 mm avec les poses `ba`. Le cercle 3D n'y est pour rien : avec les
  caméras exactes, hauteur − épaisseur ≤ 0,18 mm sur les 30 séries.

**Corrections.**

- Le plan de table exclut les points dont la majorité des observations tombent dans le masque de la référence
  (`sparse_points` prend plusieurs dossiers de masques).
- La limite passe de 0,2 à **1,0 mm**. Aucun lien entre |hauteur − épaisseur| et σ, `rms_rel` ou le nombre de
  points de table : une limite relative n'aurait rien à suivre. Avec les caméras exactes, les 30 séries passent
  (écart max 0,18 mm pour la pièce, 0,04 mm pour la carte).
- `scale.json` → `ba_warnings` : codes des avertissements de `ba_report.json`, pris en compte dans
  `checks_ok`. `ba_report.json` entre dans le hash d'entrée de l'étape.

**Résultats (étape complète).** « Avant » : entrée `6e542c6`.

| Série | Centres (mm) | Erreur d'échelle | σ | Hauteur − épaisseur avant | après | Inclinaison | Contrôles | Avertissements `ba` |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `Pony__base` | 0,35 | +0,15 % | 0,30 % | −0,06 | −0,06 | 0,13° | ok | — |
| `Ecoforms__flip` | 0,41 | −0,62 % | 0,55 % | −0,37 | −0,15 | 0,78° | ok | — |
| `OXO__noref` | 0,48 | −0,03 % | 0,10 % | +0,04 | +0,06 | 0,06° | ok | — |
| `Razer__flip` | 0,50 | +0,08 % | 0,33 % | −0,14 | −0,13 | 0,50° | ok | — |
| `Hammer__flip` | 0,59 | +0,31 % | 0,34 % | +0,03 | +0,05 | 0,41° | ok | — |
| `Pony__incomplete` | 0,66 | +0,37 % | 0,33 % | −0,67 | −0,50 | 0,54° | ok | — |
| `Caddy__base` | 0,89 | +1,17 % | 0,51 % | −1,02 | +0,09 | 0,23° | ok | — |
| `Mug__base` | 0,89 | −0,01 % | 0,06 % | +0,08 | +0,15 | 0,04° | ok | — |
| `Shurtape__base` | 0,96 | +1,19 % | 1,08 % | −2,54 | **+11,03** | **21,4°** | signalé | — |
| `Razer__noir` | 1,06 | −0,01 % | 0,37 % | −2,20 | −0,44 | 0,72° | ok | — |
| `Shurtape__incomplete` | 1,07 | +0,68 % | 0,75 % | −1,73 | −0,03 | 0,54° | ok | — |
| `Logitech__base` | 1,08 | −0,03 % | 0,06 % | −0,51 | +0,02 | 0,05° | ok | — |
| `Pony__noir` | 1,08 | +0,79 % | 0,48 % | −1,18 | −0,51 | 0,15° | ok | — |
| `OXO__noir` | 1,16 | −0,10 % | 0,08 % | −0,48 | −0,48 | 0,36° | ok | — |
| `Razer__base` | 1,21 | −0,12 % | 0,45 % | −2,60 | −0,91 | 0,92° | ok | — |
| `Teapot__noir` | 1,35 | +0,02 % | 0,29 % | +0,02 | +0,03 | 0,22° | ok | — |
| `Teapot__incomplete` | 1,39 | −0,33 % | 0,32 % | −0,05 | +0,04 | 0,74° | ok | — |
| `OXO__base` | 1,40 | +0,05 % | 0,14 % | −0,56 | +0,06 | 0,62° | ok | — |
| `Teapot__base` | 1,64 | +0,86 % | 0,69 % | +0,97 | +0,98 | **1,29°** | signalé | — |
| `Hammer__base` | 1,95 | +1,70 % | 0,92 % | −0,56 | −0,62 | 0,77° | ok | — |
| `Hammer__noir` | 1,96 | −0,06 % | 0,32 % | −2,04 | **−1,88** | 0,14° | signalé | — |
| `Mug__noref` | 2,08 | −0,18 % | 0,11 % | −0,70 | **−6,87** | **3,12°** | signalé | — |
| `Logitech__noref` | 3,05 | −0,10 % | 0,09 % | −0,76 | −0,80 | 0,21° | **ok** | — |
| `Mug__noir` | 3,14 | −0,68 % | 0,22 % | −0,61 | −0,35 | 0,36° | signalé | weak_views |
| `Ecoforms__base` | 3,98 | refusé | — | — | — | — | — | — |
| `Logitech__noir` | 4,75 | +0,11 % | 0,65 % | −0,73 | −0,60 | 0,79° | **ok** | — |
| `Caddy__noir` | 8,61 | +0,96 % | 0,28 % | +0,20 | +0,22 | 0,37° | signalé | pose_drift |
| `Shurtape__noir` | 13,09 | +2,16 % | 0,37 % | −2,07 | +0,14 | 0,88° | signalé | weak_views, pose_drift, focal_mismatch |
| `Caddy__incomplete` | 17,01 | refusé | — | — | — | — | — | — |
| `Ecoforms__noir` | 33,15 | **refusé** (`rms_rel` 18 %) | — | — | — | — | — | weak_views, pose_drift |

| | Avant (`6e542c6`) | Après (`758cc4c`) |
|---|---:|---:|
| Poses justes (centres < 2 mm) : tous contrôles ok | 7 / 21 | **18 / 21** |
| Erreur d'échelle dans 2σ : tous contrôles ok (critère : ≥ 90 %) | — | **19 / 23 (83 %)** |
| Poses fausses (centres ≥ 2 mm) : refusées ou signalées (critère : toutes) | 7 / 9 | **7 / 9** |

- **Critère non atteint.** Ses deux moitiés se contredisent ici : `Mug__noref`, `Logitech__noref` et
  `Logitech__noir` ont des poses fausses (2 à 4,75 mm) mais une erreur d'échelle dans 2σ (−0,18 à +0,11 %).
  Les signaler compte comme fausse alerte dans la première moitié. Même en signalant les 9 séries aux poses
  fausses, on plafonne à 20 / 23 (87 %).
- Restent non signalées : `Logitech__noref` et `Logitech__noir`, avec des erreurs d'échelle de −0,10 % et +0,11 %.
  Leur hauteur (−0,80 et −0,60 mm) se confond avec celle des séries justes (jusqu'à −0,91 mm). Il faudrait une
  limite de 0,5 mm pour les signaler, au prix de 10 fausses alertes sur 23.
- Fausses alertes restantes :
  - `Hammer__noir` (−1,88 mm, centres 1,96 mm, à la limite des poses fausses) ;
  - `Teapot__base` (inclinaison 1,29°) ;
  - `Mug__noref` (poses fausses, signalée à raison) ;
  - `Shurtape__base` : plan de table faux (normale à 22°).
- **Table unie.** `Shurtape__base` et `Mug__noref` (table `plain`) n'ont presque pas de points de table hors de
  la pièce (117 et 371 retenus). Les points de la pièce faisaient tenir le plan, à la mauvaise hauteur. Sans eux,
  RANSAC retient un plan faux. Les contrôles le signalent, mais le « haut » des vues en dépend : l'erreur
  d'échelle de `Shurtape__base` passe de +0,45 % à +1,19 %. Repli sur le plan de la référence : #110.
- `Ecoforms__noir` (+5,29 % avant) est désormais refusée par le garde-fou. Le plan de table corrigé change le
  « haut » des vues, ce qui fait ressortir l'incohérence des poses.

## Pour la décision M4

- Avec les caméras exactes, le σ médian de la pièce (0,27 %) dépasse déjà la cible de 0,15 % ; celui de la
  carte (0,05 %) est sous la cible de 0,08 %.
- Avec les poses `ba` corrigées et le plan de table sans la référence (`758cc4c`), l'étape aboutit sur les 21
  séries aux poses justes :
  - pièce : |erreur| médiane 0,35 % (P90 1,18 %, max 1,70 %), 5 séries sur 16 sous 0,15 %, 15 sur 16 dans 2σ ;
  - carte : médiane 0,03 %, 4 sur 5 sous 0,08 % (la 5ᵉ à 0,10 %).
  
  Ces chiffres servent de base à la décision (#51).

## Suites

- M7 : σ_scale = (σ_s / s) · ‖p − origine‖ par sommet ; réétalonner le plancher sur plus de séries et sur
  DTU / MobileBrick (68 % ± 10 % dans 1σ).
- Ajouter l'incertitude des poses à σ une fois `ba` corrigé (le jackknife ne la voit pas).
