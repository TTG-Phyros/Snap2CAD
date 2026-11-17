# M4 — Plan de table et repère monde (`scale.table_plane`, `scale.world_frame`, `scale.run`)

Mesures brutes : [`scale.json`](scale.json) (entrée du commit `59c370c`, clé `fit3d.step`), produites par
`python -m tools.eval.scale` (30 séries synthétiques, caméras et points de `ba`). Vérité : la table est le plan
z = 0 ; les poses `ba` sont alignées sur celles du rendu par similitude des centres des caméras.

## Méthode

- **Points :** points 3D de `sparse/`. Un point est « objet » si plus de la moitié de ses observations tombent
  dans le masque objet (`segment`). Les autres servent au plan (table et référence).
- **RANSAC :** plans par 3 points, seuil = 0,3 % de la distance médiane caméras → points (≈ 1 mm à 30 cm),
  2 000 tirages. Un plan qui laisse des caméras des deux côtés est écarté : il écarte les murs et les faces
  verticales. Puis moindres carrés (3 tours) sur les points retenus. Échec sous 30 points retenus.
- **Normale :** orientée vers les caméras (+Z).
- **« Haut » par vue** pour `fit_view` : normale projetée au point de la table vu au centre du masque de la
  référence. Remplace la normale exacte des évaluations précédentes.
- **Repère monde** (`world_from_sparse`, similitude vers les mm) : origine = centre du cercle ou du rectangle 3D
  projeté sur la table ; X = grand axe des points objet projetés sur la table, orienté vers l'objet.
- **Échelle :** diamètre (pièce) ou grand côté (carte : 0,04 % contre 0,29 % pour le petit côté, voir
  [M4-fit3d.md](M4-fit3d.md)).

## Résultats

Plan de table seul (sans l'ajustement de la référence) :

| Erreur de normale | Séries |
|---|---:|
| < 0,5° | 8 |
| 0,5 à 2° | 7 |
| 2 à 8° | 8 |
| Plan refusé (< 30 points, ou aucun plan sous toutes les caméras) | 7 |

- Les séries refusées ont 17 à 306 points dans `sparse/` : `ba` ne garde presque rien.
- Les meilleures séries (0,09° à 0,34°) sont celles où `ba` est juste : `Pony_C_Clamp__base`,
  `Razer__flip`, `Hammer__flip`, `Mug__base`, `Logitech__base`, `Caddy__base`. Les erreurs suivent celles des
  poses `ba` ([M4-fit3d.md](M4-fit3d.md)), pas le RANSAC.
- Le plan est en moyenne 0,5 à 3 mm au-dessus de z = 0 sur ces séries. Cet écart contient l'erreur de la
  similitude d'alignement (centres des caméras à ≈ 300 mm) ; il ne touche pas l'échelle.

Étape complète (`scale.run`) : 5 séries sur 30 aboutissent, comme avec la normale exacte (5 sur 30). Les autres
échouent sur le garde-fou de cohérence entre vues (poses `ba` fausses).

| Série | Erreur d'échelle | Normale | Origine (x, y) | Hauteur référence | Axe X |
|---|---:|---:|---:|---:|---:|
| `Razer__flip` (pièce 1 €) | +0,04 % | 0,09° | 0,4 mm | 2,22 mm (2,33) | 15,7° |
| `Logitech__base` (carte) | +1,22 % | 0,29° | 0,4 mm | 0,11 mm (0,76) | — (debout) |
| `Ecoforms__noir` (pièce 1 €) | +0,86 % | 2,77° | 20,6 mm | 1,99 mm (2,33) | — (debout) |
| `Caddy__incomplete` (pièce 2 €) | +1,85 % | 0,61° | 1,8 mm | 1,78 mm (2,20) | 0,9° |
| `Teapot__noir` (pièce 2 €) | +1,97 % | 0,45° | 3,5 mm | 2,29 mm (2,20) | 33,7° |

- Avec de bonnes poses (`Razer__flip`), le repère est juste : origine à 0,4 mm, hauteur de la face haute à
  0,1 mm de l'épaisseur réelle, échelle à 0,04 %.
- La hauteur de la référence au-dessus du plan est un contrôle de cohérence gratuit (épaisseur connue) : à
  utiliser dans la tâche « σ d'échelle et contrôles de cohérence ».
- Axe X : bien défini pour un objet allongé (`Caddy`, 0,9°). Moins pour une souris retournée ou une théière
  (anse et bec) ; c'est une convention d'affichage, pas une mesure.

## Suites

- **M3 (bloquant) :** corriger `ba` sur le synthétique. Le plan et l'échelle en dépendent directement.
- **Contrôles :** hauteur de la référence vs son épaisseur, nombre de points du plan, rms du plan.
- Le plan pourrait contraindre la normale du cercle 3D (non fait : la normale libre est déjà à 0,1° avec les
  caméras exactes).
- Temps : 4 à 22 s par série pour toute l’évaluation (processeur), surtout les contours en pleine résolution.
