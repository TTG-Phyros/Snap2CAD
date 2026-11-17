# M4 — Cercle ou rectangle 3D sur plusieurs vues (`scale.fit_circle`, `scale.fit_rectangle`)

Mesures brutes : [`scale.json`](scale.json) (entrée du commit `8e7ab1f`, clé `fit3d`), produites par `python -m tools.eval.scale`
(30 séries synthétiques). Entrées : les ajustements par vue de `scale.fit_view` ([M4-contour.md](M4-contour.md)),
avec la normale de table exacte. Deux jeux de caméras :
- **exactes** : poses et intrinsèques du rendu, en mm. On isole l'ajustement 3D et le contour ;
- **`ba`** : sortie de l'étape `ba` (M3) sur ces séries, avec sa distorsion. La vérité d'échelle est la similitude
  qui aligne les centres des caméras `ba` sur ceux du rendu.

Métrique : erreur relative de l'échelle (mm par unité) par série. Pièce : diamètre. Carte : grand puis petit côté.

## Méthode

- **Pièce :** cercle 3D (centre, normale, rayon). Les résidus sont la distance de Sampson de 64 points de chaque
  ellipse observée à la conique projetée, ce qui donne un poids égal par vue.
- **Carte :** rectangle 3D (centre, rotation, deux côtés libres). Les résidus sont l'écart entre les coins
  projetés et les coins observés. Le sens et le point de départ des coins sont appariés sur la projection
  courante.
- **Initialisation :**
  - centre : point le plus proche des rayons passant par les centres des ellipses, ou par l'intersection des
    diagonales pour la carte (projection exacte du centre) ;
  - normale : direction moyenne vers les caméras ;
  - taille : demi-axe × profondeur / focale ; pour la carte, les coins de la plus grande vue remontés sur ce
    plan.
- **Optimisation :** Levenberg-Marquardt en numpy (6 ou 8 paramètres normalisés), perte de Huber à 1 px.
- **Rejet des vues :** on écarte les vues dont le rms dépasse max(0,5 px, 3 × la médiane), puis on refait
  l'ajustement jusqu'à stabilité. Minimum : 5 vues pour une pièce, 3 pour une carte.
- **Garde-fou :** échec si le rms médian des vues gardées dépasse 1,5 px. Le rejet relatif ne voit rien
  quand toutes les vues sont fausses ensemble. Avec les caméras exactes, ce rms médian va de 0,16 à 1,1 px.

## Résultats

| Référence | Caméras | Séries ajustées | Erreur médiane | Max | Biais moyen | Cible M4 |
|---|---|---:|---:|---:|---:|---:|
| Pièce | exactes | 21 / 21 | 0,19 % | 0,87 % | +0,05 % | 0,15 % |
| Carte, grand côté | exactes | 9 / 9 | 0,04 % | 0,09 % | −0,03 % | 0,08 % |
| Carte, petit côté | exactes | 9 / 9 | 0,29 % | 0,58 % | −0,29 % | 0,08 % |
| Pièce | `ba` | 4 / 21 | 1,36 % | 1,97 % | +1,18 % | 0,15 % |
| Carte (grand, petit) | `ba` | 1 / 9 | 1,22 %, 0,99 % | — | — | 0,08 % |

Avec les caméras exactes, l'ajustement 3D est juste :
- le cercle est à 2,19 mm de hauteur pour 2,20, normale inclinée de 0,1° ;
- les tests à données idéales donnent < 0,05 %.

Les erreurs restantes viennent du contour par vue :
- **Pièce :** l'erreur est celle de la moyenne des vues, avec un biais propre à chaque série (petit axe surtout).
  Les 12 séries `coin2` sortent toutes trop petites (+0,02 à +0,35 %).
- **Carte :** les vues sont en moyenne 0,18 % trop grandes (biais déjà relevé dans [M4-contour.md](M4-contour.md)).
  En 3D, ce biais retombe presque entièrement sur le petit côté, qui dépend de la position des grands bords.
  Le grand côté passe la cible.
- **Tri des vues :** les vues ratées (erreur de taille > 2 %) sont presque toutes écartées. Il en reste 10
  gardées pour la pièce et aucune pour la carte. En contrepartie, 40 vues correctes mais moins précises sont aussi
  écartées.

## Constat bloquant : `ba` sur le synthétique

> Cause trouvée et corrigée en grande partie (#97) : voir [M3-ba-synth.md](M3-ba-synth.md). Chiffres ci-dessous : état avant correction.

Les poses de `ba` sur ces 30 séries sont bien moins bonnes que sur DTU et MobileBrick (M3) :

| Erreur médiane des centres de caméra | Séries |
|---|---:|
| < 2 mm | 7 |
| 2 à 10 mm | 10 |
| 10 à 50 mm | 7 |
| > 50 mm (focale fausse jusqu'à 135 % pour 5 d'entre elles) | 6 |

- La reprojection médiane reste souvent < 1 px. Le critère M3 ne détecte donc pas ces échecs.
- Les erreurs d'échelle avec `ba` suivent celles des poses. Même les séries à moins de 2 mm donnent 0,05 à 2 %,
  avec des résidus de 1 à 3 px au lieu de 0,3 px.
- Le garde-fou de 1,5 px fait échouer 25 séries sur 30 au lieu de rendre une échelle fausse (jusqu'à −87 %
  sans lui).
- La décision M4 ne peut pas être prise tant que `ba` n'est pas fiable sur le synthétique.

## Suites

- **M3 :** comprendre l'échec de `ba` sur le synthétique (table à motif, objet petit dans l'image, focale non
  bloquée ?).
- **Contour de la carte :** biais de +0,18 % (petit côté en 3D).
- **Plan de table (tâche suivante) :** fournira la direction « haut » par vue, encore prise sur la vérité
  terrain ici. Il pourra aussi contraindre la normale du cercle.
- Temps : 0,1 à 0,5 s par ajustement 3D (processeur).
