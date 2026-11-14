# M2 — Détourage SAM 2.1 en mode vidéo (sans repli 3D)

Mesures brutes : [`segment.json`](segment.json), produites par `python -m tools.eval.segment`.
Conditions : étape `segment` (SAM 2.1 `base_plus`, profil 6 Go), 20 vues réparties sur chacune des 30 séries
synthétiques (M1b), triées par heure EXIF, RTX 3060 Laptop, Windows 11.
Clics simulés sur la 1ʳᵉ vue où la cible est visible : un point au cœur, puis des corrections jusqu'à
IoU ≥ 0,95 (au plus 5). IoU comptée sur les vues où la cible occupe ≥ 0,01 % de l'image ; « perdue » = IoU < 0,5.

| Cible | IoU moyenne | Séries ≥ 0,9 | Séries sans vue perdue | Clics (moyenne) |
|---|---:|---:|---:|---:|
| Objet | **0,927** | 24 / 30 | 25 / 30 | 1,7 |
| Référence (pièce, carte) | 0,781 | 13 / 30 | 9 / 30 | 2,2 |

Critère du plan (IoU ≥ 0,9, aucune perte de l'objet sur 20 vues) : **pas encore atteint**.

Observations :
- Sur la vue cliquée, l'IoU est toujours ≥ 0,94 (objet) : les échecs viennent de la propagation, pas du clic.
- Avec un seul clic, SAM choisit souvent une partie de l'objet (ouvre-boîte 0,03, souris retournée 0,02) :
  l'interface doit montrer le masque et inviter à corriger avant de propager.
- Objet perdu sur 5 séries : souris `flip` (13 vues), marteau `noir` (7), mug `noref` (5),
  enceinte `noref` (3), pot `noir` (2). Vues très espacées (jusqu'à ≈ 55° d'azimut entre deux vues) : c'est le risque R3.
- La référence, petite (≈ 0,1 à 1 % de l'image), se perd bien plus souvent et ne revient pas toujours.
- Suite : repli guidé en 3D (clic remonté par la profondeur VGGT et reprojeté dans chaque vue), puis nouvelle mesure.
- Temps de l'étape : ≈ 11 s par série de 20 vues (médiane), chargement du modèle compris.

## Avec correction manuelle (clics sur les vues fautives)

L'étape `segment` accepte des clics positifs et négatifs sur n'importe quelle photo : corriger, c'est ajouter
des clics sur la vue fausse et relancer l'étape (≈ 9 s pour 20 vues). L'évaluation simule l'utilisateur :
après chaque passage, il corrige la pire vue sous IoU 0,8 où la cible est assez visible pour cliquer
(≥ 2000 px), avec les mêmes clics que sur la 1ʳᵉ vue, au plus 3 vues par cible. Mêmes conditions que ci-dessus.

| Cible | IoU moyenne | Séries ≥ 0,9 | Séries sans vue perdue | Vues corrigées (moyenne) |
|---|---:|---:|---:|---:|
| Objet | 0,927 → **0,982** | 24 → **30** / 30 | 25 → **30** / 30 | 0,4 |
| Référence (pièce, carte) | 0,781 → 0,947 | 13 → 28 / 30 | 9 → 25 / 30 | 1,1 |

Critère du plan pour l'objet (IoU ≥ 0,9, aucune perte sur 20 vues) : **atteint avec correction**, au plus 3 vues
corrigées par série (souris `flip` : 1 vue suffit pour 13 vues perdues ; marteau `noir` : 3 vues).

Observations :
- Une seule vue corrigée rattrape souvent toute la série : le clic sert de mémoire à la propagation dans les deux sens.
- Référence encore perdue sur 5 séries. Sur 3 d'entre elles, une seule vue est perdue et la pièce y est trop petite
  pour un clic simulé (< 2000 px). Sur les 2 autres (`Ecoforms…flip`, `Pony_C_Clamp…base`), 3 corrections ne suffisent pas.
  L'échelle demande la référence sur ≥ 5 vues (plan §1), pas sur toutes : ces vues seront écartées en M4.
- La simulation est optimiste sur le choix de la vue (elle connaît la vérité terrain) ; un utilisateur la voit à l'écran.
