# M4 — Contour de la référence par vue (`scale.fit_view`)

Mesures brutes : [`scale.json`](scale.json), produites par `python -m tools.eval.scale` (30 séries synthétiques).
Conditions : masques de référence de l'étape `segment` (SAM 2.1, après correction simulée), caméras exactes
(k1 compris) et normale de la table projetée exacte. On isole ainsi l'erreur du contour ; celle des poses (`ba`)
et du plan de table viendra avec l'ajustement 3D.

Métrique : erreur relative de taille de l'arête haute par vue (pièce : grand demi-axe, non raccourci ; carte :
moyenne des 4 côtés). Une vue « ratée » dépasse 2 % ou n'a pas pu être ajustée.

| Référence | Vues | Réussies | Erreur médiane | P90 | Biais moyen |
|---|---:|---:|---:|---:|---:|
| Pièce (1 € et 2 €) | 334 | 238 (71 %) | 0,22 % | 0,76 % | +0,00 % |
| Carte ID-1 | 155 | 93 (60 %) | 0,13 % | 0,40 % | +0,18 % |

Causes des échecs :

| Cause | Pièce | Carte |
|---|---:|---:|
| Référence partiellement cachée par l'objet (< 97 % visible) | 23 | 25 |
| Flou de bougé | 19 | 6 |
| Masque SAM faux (IoU < 0,9) | 4 | 0 |
| Détecteur (tranche pâle confondue avec la table, carte claire sur carrelage clair) | 50 | 31 |

Méthode et constats :
- **On vise l'arête de la face haute, pas la silhouette.** Côté caméra, la silhouette passe par l'arête basse,
  et la tranche sombre plus l'ombre de contact la décalent encore d'environ 1 px vers l'extérieur. Ajuster
  la silhouette donnait +0,8 px de biais, soit environ 1 % sur une pièce de 150 px. L'arête haute, elle, se
  projette exactement en ellipse (ou en quadrilatère).
- Les bords sont cherchés le long de rayons normaux à la forme initiale (ajustée au masque SAM). On utilise
  la dérivée couleur le long du rayon, affinée par une parabole. Sur un bord idéal, le biais est de 0,02 px.
- Pièce : la frappe donne des bords intérieurs plus forts que le contour, et la tranche (2,2 mm) est
  réfléchissante.
  - Premier passage sur le bord le plus extérieur. Côté caméra, l'arête basse est modélisée comme l'arête
    haute translatée de t le long de la normale projetée.
  - La silhouette seule ne permet pas de séparer t de la taille de l'ellipse. t vient donc de l'épaisseur
    connue : t = (épaisseur / rayon) · a · √(1 − (b/a)²).
  - Ensuite, on prend le bord le plus proche de l'ellipse, sur des rayons de ±3 puis ±1,5 px.
- Carte : tranche fine (0,76 mm).
  - Au premier passage, côté caméra, on prend le bord qui précède la silhouette dans une fenêtre bornée
    par l'épaisseur projetée.
  - Ensuite, on prend le bord le plus proche des droites. Les 12 % de chaque côté près des coins arrondis
    sont écartés.
- La direction « haut » (normale de la table projetée) doit être la vraie. Avec le haut de l'image, une
  pièce excentrée mélange les arêtes haute et basse sur le côté (erreur de plusieurs px). En production,
  elle viendra du plan de table (tâche suivante) et des poses de `ba`.
- Le rms interne de l'ajustement (≈ 0,7 px) ne distingue pas les vues ratées. Leur rejet se fera par
  cohérence entre vues dans l'ajustement 3D.
- Ordre de grandeur pour la décision M4 (cible : 0,15 % pièce, 0,08 % carte) : la dispersion par vue est
  aléatoire sur la pièce (biais ≈ 0). La moyenne de 10 à 15 vues réussies tomberait vers 0,07 à 0,1 %. La
  carte garde un biais de +0,18 %, qu'il faudra comprendre (ombre de contact résiduelle côté caméra ?).
- Temps : 0,1 à 0,3 s par vue (processeur).
