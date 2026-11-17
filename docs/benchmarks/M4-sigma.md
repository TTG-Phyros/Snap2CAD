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

Étape complète (poses `ba`), les 5 séries qui aboutissent :

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

## Pour la décision M4

- Avec les caméras exactes, le σ médian de la pièce (0,27 %) dépasse déjà la cible de 0,15 % ; celui de la
  carte (0,05 %) est sous la cible de 0,08 %.
- La décision attend toujours la correction de `ba` sur le synthétique.

## Suites

- M7 : σ_scale = (σ_s / s) · ‖p − origine‖ par sommet ; réétalonner le plancher sur plus de séries et sur
  DTU / MobileBrick (68 % ± 10 % dans 1σ).
- Ajouter l'incertitude des poses à σ une fois `ba` corrigé (le jackknife ne la voit pas).
