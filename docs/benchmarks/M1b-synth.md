# M1b — Jeu synthétique : premier lot et contrôle par VGGT

Générateur : `tools/synth` (Blender 4.2 en module `bpy`, Cycles OptiX, 32 échantillons, débruitage OptiX).
Mesures VGGT : [`poses.json`](poses.json) (entrée du 2026-11-14, commit `a5cc7f7`), produites par
`python -m tools.eval.poses synth:<série> ...` ; mêmes conditions que [M1](M1-poses.md) (VGGT-1B bf16,
20 vues réparties, RTX 3060 Laptop, alignement Umeyama sur les centres).

## Premier lot

10 objets Google Scanned Objects (CC-BY 4.0, Gazebo Fuel) de 10 à 30 cm × 3 variantes = 30 séries, 16 à 40 vues,
1,2 Go. Variantes : `base` et `noir` (matériau noir mat) pour chaque objet, puis `incomplete` (secteur de 100° sans
vue), `flip` (objet retourné) ou `noref` (référence hors cadre sur 6 vues) en alternance.
Références : pièce de 2 €, de 1 €, carte ID-1 ; tables bois, uni, motif.
Smartphone simulé : focale 24 – 26 mm équivalent 35 mm, distorsion radiale |k1| ≤ 0,015, flou de bougé jusqu'à 12 px,
bruit, JPEG qualité ≥ 85 avec EXIF (focale, date à la sous-seconde).

Sorties par série (`datasets/synth/<série>/`) : images, étiquettes (table / objet / référence), profondeur (mm × 20),
`cameras.npz` (OpenCV, mm, même format que les adaptateurs DTU / MobileBrick), `mesh.ply` (monde, mm), `scene.json`
(cotes hors-tout, pose de la référence, réglages par vue).

Auto-contrôle à la génération : sommets du maillage projetés sur le masque objet (≥ 92,6 % sur toutes les vues) et
comparés à la carte de profondeur (≤ 4,3 % devant la surface rendue). Seuil « sur la surface » ramené de 10 % à 5 % :
les objets creux (pot, range-stylos) ont jusqu'à 92 % de sommets sur des parois cachées.

## VGGT seul sur le lot

| Variante | Séries | Rotation médiane (médiane) | Pire série |
|---|---:|---:|---:|
| base | 10 | 1,05° | 1,59° (souris) |
| noir | 10 | 1,35° | 1,92° (ruban adhésif) |
| incomplete | 4 | 1,24° | 2,18° (serre-joint) |
| flip | 3 | 1,04° | 1,40° |
| noref | 3 | 1,38° | 1,41° |
| **Toutes** | **30** | **1,15°** | 2,18° |

Centre : 4,6 – 16 mm (médiane 8,6 mm). Focale VGGT surestimée de 1,8 à 15 % (médiane + 7,6 %). 18 à 24 s par série.

Critère M1b atteint : la vérité terrain est relue par l'adaptateur `synth` et s'aligne avec VGGT à ≈ 1° près, comme
MobileBrick (1,07°) ; une erreur de convention (axes, mm, pose inversée) donnerait des dizaines de degrés.

Observations :
- Le noir mat coûte ≈ 0,3° : peu de texture sur l'objet, VGGT s'appuie sur la table et la référence.
- La focale est plus surestimée que sur MobileBrick (+ 3 %) : grand-angle 24 mm et distorsion k1 non modélisée par
  VGGT. Confirme que l'ajustement M3 doit libérer la focale (et k1).

## Raccourcis connus du générateur

- Pièces à tranche lisse (sans cannelures ni gravure de tranche).
- JPEG seulement (pas de HEIC), un seul modèle de distorsion (SIMPLE_RADIAL).
- `scene.json` donne les cotes hors-tout, pas encore les diamètres ni entraxes (à ajouter avec M7).
- Objet retourné posé par rotation de 180° autour de x, sans simulation physique de l'appui.
