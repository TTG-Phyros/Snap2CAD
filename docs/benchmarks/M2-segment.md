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
