# M4 — Décision : erreur d'échelle sur le synthétique (#51)

Mesures brutes : entrée `6e542c6` de [`scale.json`](scale.json), produite par `python -m tools.eval.scale` (30 séries
synthétiques, 20 vues réparties, RTX 3060 Laptop, Windows 11). Détail par série et méthode :
[M4-sigma.md](M4-sigma.md). Erreur d'échelle relative |s_est / s_vrai − 1| (plan §9.4) ; carte : grand côté
(échelle retenue par l'étape `scale`).

Deux cas :
- **Caméras exactes :** poses et intrinsèques de la vérité terrain. Seule l'erreur du contour compte : c'est la
  borne basse de la méthode.
- **Poses `ba` :** étape `scale` complète (poses `ba` corrigées par #97, plan de table RANSAC, garde-fou #101).
  Séries aux poses justes uniquement (centres < 2 mm, 21 sur 30). Les 9 autres sont refusées ou signalées
  ([M4-sigma.md](M4-sigma.md)).

## Résultats

| Référence | Cas | Séries | \|erreur\| médiane | P90 | Max | Sous la cible | Sur 30 cm (médiane / P90) |
|---|---|---:|---:|---:|---:|---:|---:|
| Pièce (cible < 0,15 %) | caméras exactes | 21 | **0,19 %** | 0,35 % | 0,87 % | 8 / 21 | 0,57 / 1,06 mm |
| | poses `ba` | 16 | **0,38 %** | 1,00 % | 1,71 % | 4 / 16 | 1,13 / 3,00 mm |
| Carte (cible < 0,08 %) | caméras exactes | 9 | **0,040 %** | 0,084 % | 0,089 % | 7 / 9 | 0,12 / 0,25 mm |
| | poses `ba` | 5 | **0,029 %** | 0,078 % | 0,095 % | 4 / 5 | 0,09 / 0,23 mm |

Par pièce (médiane, caméras exactes / poses `ba`) : 1 € 0,19 % / 0,27 % (9 / 7 séries), 2 € 0,19 % / 0,45 %
(12 / 9 séries).

## Critères

- **Pièce < 0,15 % : non atteint.** Même avec les caméras exactes, l'erreur médiane (0,19 %) et le σ médian
  (0,27 %) dépassent la cible. La limite vient du contour : ≈ 0,2 px d'erreur sur un rayon de 60 à 180 px. Avec
  les poses `ba`, les écarts résiduels de pose (≈ 1 à 3,5 px, [M4-sigma.md](M4-sigma.md)) doublent l'erreur
  médiane. Une meilleure `ba` (#100) ne suffira donc pas à atteindre la cible.
- **Carte < 0,08 % : atteint en médiane** (0,03 à 0,04 %), sur 11 séries sur 14. Les 3 dépassements restent
  ≤ 0,095 % (0,29 mm sur 30 cm). La carte est ≈ 3 fois plus grande que la pièce dans l'image, et ses 4 coins
  la rendent peu sensible aux erreurs de pose (≤ 0,12 % avec 1 mm et 0,3° de bruit par vue).

## Décision

La cible n'est pas atteinte avec la pièce : on applique la décision déjà validée (plan §2 et §10, M4).

- **Carte ID-1 obligatoire** comme référence d'échelle. La pièce ne suffit pas seule (0,6 à 1,1 mm d'erreur
  médiane sur 30 cm).
- **Cotes au pied à coulisse obligatoires** pour atteindre ±0,2 mm sur les cotes. Même avec la carte, l'erreur
  d'échelle P90 (≈ 0,25 mm sur 30 cm) consomme à elle seule le budget de ±0,2 mm : le recalage par au moins
  2 cotes (M7) reste nécessaire.

Conséquences sur la suite (à reprendre dans les tickets concernés) :

- M7 (#67) : le recalage d'échelle par 2 cotes devient le chemin normal, pas une option.
- M8 (#75) : l'assistant de prise de vue exige la carte, visible sur au moins 5 photos, et la saisie de cotes
  au pied à coulisse avant d'afficher des cotes finales.

## Limites

- Synthétique uniquement : 9 séries avec carte, dont 5 aux poses `ba` justes. L'échantillon de la carte est petit.
- **À revalider sur photos réelles** (plan §9.4, validation finale) : contour de la carte réelle (coins usés,
  reflets, relief des caractères), pièce réelle (tranche striée, usure).
- Les contrôles de `scale` donnent beaucoup de fausses alertes avec les poses `ba` (#102). Cela ne change pas
  la décision.
