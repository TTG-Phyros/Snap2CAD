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
  - hauteur de la face haute au-dessus du plan de table − épaisseur de la référence : ≤ 0,2 mm ;
  - angle entre la normale de la référence et celle de la table : ≤ 1° ;
  - carte : échelle du petit côté vs grand côté : ≤ 1 % (biais du petit côté jusqu'à 0,57 % avec les caméras
    exactes, [M4-fit3d.md](M4-fit3d.md)).

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
  −2,6 mm sur `Razer__base`, erreur −0,10 %) : 7 séries justes sur 21 seulement passent tous les contrôles. Suivi
  dans un ticket à part.
- `Ecoforms__noir` (+5,29 %) et `Caddy__noir` (+0,94 %) passent les contrôles de `scale` : seul `ba` les signale.

## Pour la décision M4

- Avec les caméras exactes, le σ médian de la pièce (0,27 %) dépasse déjà la cible de 0,15 % ; celui de la
  carte (0,05 %) est sous la cible de 0,08 %.
- Avec les poses `ba` corrigées, l'étape aboutit sur les 21 séries aux poses justes. Pièce : |erreur| médiane
  0,38 %, 4 séries sur 16 sous 0,15 %. Carte : médiane 0,03 %, 4 sur 5 sous 0,08 % (la 5ᵉ à 0,095 %).
  Ces chiffres servent de base à la décision (#51).

## Suites

- M7 : σ_scale = (σ_s / s) · ‖p − origine‖ par sommet ; réétalonner le plancher sur plus de séries et sur
  DTU / MobileBrick (68 % ± 10 % dans 1σ).
- Ajouter l'incertitude des poses à σ une fois `ba` corrigé (le jackknife ne la voit pas).
