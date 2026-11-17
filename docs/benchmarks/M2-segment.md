# M2 — Détourage SAM 2.1 en mode vidéo, correction manuelle, repli guidé en 3D et repli BiRefNet

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

## Avec repli guidé en 3D (#37)

Entrée `f774e8e` de [`segment.json`](segment.json), comparée à `f8b7f3c` (sans repli). Mêmes séries, mêmes vues, même
simulation de l'utilisateur, même GPU. Le repli utilise `sparse/` de l'étape `ba` (poses corrigées par #97), déjà calculé
dans `datasets/_eval` par `tools.eval.poses --ba`.

**Méthode** (`worker/snap2cad/steps/segment.py`) :
- Les points 3D de la cible sont ceux de `sparse/` observés dans les photos fiables, et dans le masque pour plus de
  la moitié de ces observations. Au départ, seules les photos cliquées sont fiables. S'y ajoutent les photos où au moins
  80 % des projections de ces points tombent dans le masque propagé, jusqu'à stabilité.
- Une photo est perdue si moins de 50 % de ces projections tombent dans son masque. On projette tous les points,
  cachés ou non, car ils retombent tous dans la silhouette ; les seules observations ne voyaient pas les photos
  prises de l'autre côté de l'objet. La photo n'est jugée que si la cible y est entière (≥ 90 % des projections dans
  l'image) et, pour la référence, pas cachée à plus de moitié par le masque de l'objet.
- La photo la plus perdue reçoit comme invite la boîte de ces projections (1ᵉʳ – 99ᵉ centiles), puis la
  propagation est relancée, au plus 3 fois par cible. L'état SAM est remis à zéro avant chaque relance : une invite
  posée sur une photo déjà suivie ne serait qu'une correction, oubliée à la propagation suivante.
- Sans `sparse/`, l'étape reste inchangée. `ba` n'utilise pas les masques : `segment` tourne après `ba` (plan §4.2).

Avant correction manuelle (SAM et repli seuls) :

| Cible | IoU moyenne | Séries ≥ 0,9 | Séries avec une vue perdue | Vues perdues (total) |
|---|---:|---:|---:|---:|
| Objet, sans → avec repli | 0,927 → **0,958** | 24 → **27** / 30 | 5 → **2** | 30 → **7** |
| Référence, sans → avec repli | 0,781 → **0,866** | 13 → **21** / 30 | 21 → **12** | 92 → **45** |

Après correction manuelle simulée (au plus 3 vues par cible) :

| Cible | IoU moyenne | Séries ≥ 0,9 | Séries avec une vue perdue | Vues corrigées (moyenne) |
|---|---:|---:|---:|---:|
| Objet, sans → avec repli | 0,982 → 0,981 | 30 → 30 / 30 | 0 → 0 | 0,40 → **0,17** |
| Référence, sans → avec repli | 0,947 → **0,961** | 28 → **29** / 30 | 5 → **4** | 1,07 → **0,60** |

Les 5 séries où SAM seul perdait l'objet (IoU moyenne / vues perdues, avant correction) :

| Série | Sans repli | Avec repli | Boîtes posées |
|---|---:|---:|---:|
| souris `flip` | 0,529 / 13 | **0,972 / 0** | 2 |
| marteau `noir` | 0,714 / 7 | **0,981 / 0** | 3 |
| mug `noref` | 0,704 / 5 | **0,977 / 0** | 2 |
| enceinte `noref` | 0,712 / 3 | 0,673 / 5 (**régression**) | 3 |
| pot `noir` | 0,826 / 2 | 0,826 / 2 (repli non déclenché) | 0 |

Observations :
- Le repli rattrape sans aucun clic 3 des 5 séries perdues, y compris la souris retournée (`flip`, poses `ba` à
  0,5 mm), que la profondeur VGGT n'aidait pas (branche WIP abandonnée).
- **Enceinte `noref` :** le haut de l'enceinte, noir et uni, n'a pas de points `ba`. La boîte s'arrête 400 px sous le
  vrai bord et SAM ne prend qu'une partie de l'objet (IoU 0,19 et 0,43 sur deux vues relancées). Une correction
  manuelle la rattrape (0,991).
- **Pot `noir` :** noir mat, 0 à 2 points `ba` par vue (1 sur la vue cliquée) : rien à remonter. Cas de BiRefNet (#39).
- La référence profite le plus du repli (vues perdues divisées par 2). Sans le test de visibilité (cible entière,
  non cachée par l'objet), des boîtes tombaient sur des vues où la pièce est cachée ou hors champ : référence du mug
  `noref` 0,45 → 0,22 au lieu de 0,81.
- Invites comparées sur 8 séries (les 5 ci-dessus et 3 où la référence se perd, avant correction) : la boîte seule
  est la meilleure. Des points positifs au cœur des projections, en plus de la boîte, aident l'enceinte (0,81 / 1)
  mais perdent la pièce du mug (0,37 / 8). Des points seuls dégradent presque tout (marteau `noir` 0,48 / 10).
- Repli et correction manuelle interagissent : sur l'enceinte `noref`, la référence est juste au 1ᵉʳ passage (0,981),
  mais redevient perdue sur une vue (0,929) après la correction de l'objet, qui change le masque de l'objet servant au
  test de visibilité.
- Temps de l'étape : 14 s par passage en médiane, contre 9 s (relances comprises).
- Les masques de `datasets/_eval` ont changé (référence surtout) : `scale.json` (`6e542c6`) a été mesuré avec les anciens.

## Avec BiRefNet en repli automatique (#39)

Mesures brutes : [`birefnet.json`](birefnet.json) (BiRefNet seul, `python -m tools.eval.birefnet`) et entrée `dd5e7fc` de
[`segment.json`](segment.json), comparée à `f774e8e` (repli 3D seul) et `f8b7f3c` (SAM seul). Mêmes séries, mêmes vues,
mêmes clics simulés, même GPU.

**BiRefNet seul** (objet saillant, 1024 px, seuil 0,5), sur les 579 vues où l'objet est visible :

| Variante | IoU moyenne | dont plus grande composante | Vues perdues | Référence prise pour l'objet | VRAM (pic) |
|---|---:|---:|---:|---:|---:|
| BiRefNet (Swin-L) | 0,966 | **0,981** | **6** | 196 vues | 3,5 Go |
| BiRefNet_lite (Swin-T) | 0,936 | 0,968 | 12 | 281 vues | 1,7 Go |

- La pièce ou la carte, posée à côté, est souvent incluse dans le masque : on ne garde que la plus grande composante
  qui ne se confond pas avec la référence vue par SAM. Les 6 vues perdues sont toutes sur l'ouvre-boîte OXO, où la carte
  est plus grande que l'objet à l'image.
- Variante retenue : **BiRefNet complet** sur les deux profils. Il est meilleur que la variante lite (6 vues perdues
  contre 12) et tient sur 6 Go, parce qu'il tourne avant que SAM soit chargé. Seule la variante complète est au
  manifeste. Le temps mesuré par vue (0,4 à 2 s) varie selon la charge de la machine : il n'est pas fiable ici.
- L'entrée de `birefnet.json` porte le commit `929a6c6` : le code mesuré était celui de `dd5e7fc`, pas encore committé.

**Méthode** (`worker/snap2cad/steps/segment.py`, `worker/snap2cad/models/birefnet.py`) :
- BiRefNet détoure chaque photo avant le chargement de SAM, pour que les deux modèles ne soient jamais en VRAM ensemble.
- On ne s'y fie que s'il s'accorde avec le masque SAM des photos cliquées (IoU ≥ 0,8), sinon il détoure autre chose que
  l'objet désigné. Il sert pour l'objet seulement : il ne voit pas la référence.
- Après le repli 3D, une photo où SAM et BiRefNet divergent (IoU < 0,5), et que la géométrie ne juge pas correcte
  (≥ 80 % des projections dans le masque), reçoit comme invite la boîte du masque BiRefNet. Celle-ci remplace une
  éventuelle boîte reprojetée. Au plus 3 relances.
- Code et poids publiés sur Hugging Face (révision `e2bf8e44`, licence MIT), SHA-256 au manifeste. Le code n'utilise
  `transformers` que pour deux classes de base : une cale les remplace. Seule dépendance ajoutée : `timm`.

**Avant correction manuelle** :

| Objet | SAM seul | + repli 3D | + BiRefNet |
|---|---:|---:|---:|
| IoU moyenne | 0,927 | 0,958 | **0,974** |
| Séries ≥ 0,9 | 24 / 30 | 27 / 30 | **29 / 30** |
| Séries avec une vue perdue | 5 | 2 | **0** |
| Vues perdues (total) | 30 | 7 | **0** |

**Après correction manuelle simulée** : objet 0,982 / 0,981 / 0,981 (30 séries ≥ 0,9, aucune perte dans les trois
cas), avec 0,40 / 0,17 / **0,10** vue corrigée par série en moyenne. Référence 0,947 / 0,961 / 0,963, séries avec une
vue perdue 5 / 4 / 3 (BiRefNet ne la traite pas : l'écart vient du masque de l'objet, qui sert au test de visibilité).

Critère du ticket (pas de régression de l'IoU moyenne ; moins de vues perdues qu'avec SAM seul) : **atteint**.
IoU avant correction 0,927 → 0,974, vues perdues 30 → 0. Après correction, l'IoU finale est égale à 0,001 près
(0,982 → 0,981).

Observations :
- BiRefNet n'a relancé que 2 séries, celles où il le fallait. Il n'y a aucune relance à tort, et le garde-fou
  d'accord avec la vue cliquée n'a coupé BiRefNet nulle part (IoU minimale 0,947).
  - **Pot `noir`** (0 à 2 points `ba` par vue, repli 3D impossible) : 0,826 avec 2 vues perdues → **0,989 avec 0**,
    une boîte.
  - **Enceinte `noref`** : 0,673 avec 5 vues perdues (boîtes 3D tronquées, voir #105) → **0,985 avec 0**, trois
    boîtes. La régression du repli 3D disparaît sur cette série.
- La seule série encore sous 0,9 avant correction, l'enceinte `noir` (0,849), n'a aucune vue perdue : ses masques
  sont partiels sans diverger assez de BiRefNet pour une relance.
- **Temps de l'étape :** 35 s en médiane au premier passage, contre 14 s avec le repli 3D seul et 9 s avec SAM seul ;
  161 s au pire (enceinte `noref` : 3 relances 3D puis 3 relances BiRefNet). BiRefNet coûte environ 1 s par photo
  sur la 3060 Laptop, et chaque relance refait toute la propagation.
