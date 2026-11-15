# Snap2CAD — Plan de développement

> Version 1.0 — 11 novembre 2026
> Statut : plan validé sur les décisions de cadrage. **M0 en cours** : socle fait, VGGT mesuré sur la RTX 4060 (N₈ = 28). N₆ reste à remesurer sur la RTX 3060 Laptop avec le plafond VRAM.
> Visuels : voir [Annexes](docs/annexes/ANNEXES.md) (architecture, pipeline, VRAM, données, jalons, validation).

---

## Sommaire

1. [Objectif et périmètre](#1-objectif-et-périmètre)
2. [Décisions validées](#2-décisions-validées)
3. [Architecture](#3-architecture)
4. [Pipeline du mode Référence](#4-pipeline-du-mode-référence)
5. [Données, formats et dossier projet](#5-données-formats-et-dossier-projet)
6. [Environnement, dépendances et intégration des dépôts](#6-environnement-dépendances-et-intégration-des-dépôts)
7. [Plateformes](#7-plateformes)
8. [Gestion de la VRAM et modes dégradés](#8-gestion-de-la-vram-et-modes-dégradés)
9. [Stratégie de validation et datasets](#9-stratégie-de-validation-et-datasets)
10. [Jalons du mode Référence](#10-jalons-du-mode-référence)
11. [Mode Éditer (phase 2)](#11-mode-éditer-phase-2)
12. [Risques et parades](#12-risques-et-parades)
13. [Organisation du dépôt et conventions](#13-organisation-du-dépôt-et-conventions)
14. [Questions encore ouvertes](#14-questions-encore-ouvertes)

---

## 1. Objectif et périmètre

Application desktop qui transforme une série de photos d'un objet en modèle 3D, selon deux modes :

| | Mode Référence (v1, prioritaire) | Mode Éditer (phase 2) |
|---|---|---|
| Sortie | Mesh à l'échelle : STL, OBJ, PLY, GLB, STEP facetté, rapport JSON | STEP + script CadQuery `.py` éditable |
| Priorité | Exactitude dimensionnelle | Topologie propre |
| Objets | Rigides et souples (les souples restent en mesh) | Rigides uniquement |

**Cibles de précision :**
- ±0,5 mm sur des pièces de 5 à 30 cm ;
- ±0,2 mm sur les cotes recalées au pied à coulisse.

Cotes prioritaires : hors-tout, diamètres, entraxes.

**Objets visés en v1 :** objets du quotidien (tabouret, bougie, casque) et pièces type porte-plaques ou répliques d'airsoft. Matières prioritaires : plastique noir mat, polymère, métal peint, souvent peu texturés. **Transparent et chromé : hors périmètre v1.**

**Règle absolue :** une zone générée par un modèle (complétion) n'est **jamais** utilisée pour une cote. Chaque point et chaque sommet porte l'étiquette `mesuré` ou `généré`.

---

## 2. Décisions validées

| Sujet | Décision |
|---|---|
| Cadre | Projet personnel. Usage de Hunyuan3D-2 autorisé selon l'utilisateur. Checkpoints : VGGT-1B, Hunyuan3D-2mv (turbo, fp16), CAD-Recode v1.5 |
| Plateformes | Windows + NVIDIA d'abord, puis Ubuntu 22.04/24.04 (AppImage), puis macOS Apple Silicon. Code agnostique au périphérique. On préfère le meilleur outil par OS plutôt qu'une stricte parité |
| GPU | 8 Go recommandés, 6 Go supportés en mode dégradé. RTX 30, 40 et 50. Pas d'AMD en v1. Pas de mode tout-processeur pour l'IA |
| Machines de dev | Portable RTX 3060 Laptop 6 Go et RTX 4060 8 Go : l'app doit fonctionner sur les deux |
| Pile logicielle | Python 3.11, torch 2.7 + CUDA 12.8 partout (version standard sur Mac) |
| Interface | Electron + React + three.js. Worker Python local en HTTP + WebSocket. PySide6 en repli si l'affichage des gros nuages pose problème |
| Distribution | Installeur léger. Poids téléchargés au premier lancement, avec reprise. **Fonctionnement entièrement hors ligne ensuite.** Pas de signature de code en v1 |
| Prise de vue | Smartphone iPhone ou Android, objectif principal uniquement. Aucune calibration demandée à l'utilisateur. Import d'un dossier de photos, puis import vidéo plus tard |
| Échelle | Objet de référence posé à plat, visible sur au moins 5 photos : pièce de 2 € (défaut), pièce de 1 €, carte ID-1 (recommandée au-delà de 15 cm). L'utilisateur clique dessus, SAM le segmente |
| Repère | Plan de la table, Z vers le haut, unités en **mm** |
| Dessous de l'objet | Deux sessions (objet retourné) puis fusion, ou une seule session avec le dessous « non couvert ». C'est l'utilisateur qui choisit |
| Couverture | Une zone est couverte si au moins 3 caméras la voient avec un angle de triangulation d'au moins 15°. Couverture suffisante au-delà de 90 % de la surface. D'abord guider la reprise de photos, la complétion n'est qu'un repli signalé |
| Incertitude | Quatre sources (confiance VGGT, nombre de vues, erreur de reprojection, écart aux cotes) combinées en mm. Export par sommet (PLY), en couleurs (GLB) et en rapport JSON |
| Cotes au pied à coulisse | Avec une seule cote : vérification. À partir de deux : recalage d'échelle par moindres carrés, résidus affichés. Désignation : deux points, cercle ajusté (diamètre), deux plans (épaisseur) |
| Outils de mesure | Distance et diamètre au minimum |
| Objets souples | Choix de l'utilisateur en v1 |
| Détourage | SAM 2.1 : clic sur la 1ʳᵉ photo puis propagation. BiRefNet automatique en repli |
| STEP en mode Référence | STEP facetté. Point2CAD abandonné en v1 |
| Support texturé, spray de scan | Support texturé seulement si nécessaire. Spray de scan conseillé mais optionnel |
| Échec de la précision au jalon M4 | Imposer la carte bancaire et des cotes au pied à coulisse |
| Fusion de sessions | Alignement manuel par 3 points accepté en repli |
| Orchestration | Un processus par étape. Dossier projet versionné par `project.json` (paramètres et hash des entrées) pour ne relancer que ce qui a changé |
| Environnement Python | Un seul environnement, repli sur deux si besoin |
| Jeu de test | Pas de photos personnelles pour l'instant : validation sur datasets publics et sur un jeu synthétique ([§9](#9-stratégie-de-validation-et-datasets)) |

---

## 3. Architecture

> Schéma : [Annexe A1 — Architecture globale](docs/annexes/ANNEXES.md#a1--architecture-globale)

### 3.1 Composants

**Application Electron**
- **Processus principal (Node) :**
  - cycle de vie de l'app ;
  - lancement et surveillance du worker Python ;
  - premier lancement : création de l'environnement Python et téléchargement des poids, avec reprise.
- **Interface (React + three.js) :**
  - écrans Projets, Import, Détourage (clics), Vue 3D, Mesures, Couverture, Rapport et exports ;
  - nuages en `THREE.Points` (BufferGeometry) ;
  - sélection de points sur le mesh avec `three-mesh-bvh` ;
  - affichage par couleurs : mesuré / généré, incertitude, couverture.

**Worker Python (`snap2cad`)**
- **API locale :** FastAPI en HTTP pour les commandes, WebSocket pour la progression et les journaux.
- **Orchestrateur :**
  - graphe des étapes ;
  - cache par hash ;
  - file d'attente, annulation, reprise après erreur.
- **Gestionnaire de périphérique :** CUDA, MPS ou processeur, avec budget mémoire et choix du mode dégradé.

**Étapes**
- Chaque étape est un module Python lançable en ligne de commande (`python -m snap2cad.steps.<nom> --project <dossier>`).
- L'orchestrateur la lance dans un **sous-processus**. Quand il se termine, toute la VRAM est libérée.
- **Une seule étape GPU à la fois.**

**Dossier projet**
- Source de vérité unique.
- L'interface ne lit que des fichiers produits par les étapes (GLB, PLY, JSON), servis en binaire par le worker.

**Magasin de modèles**
- Les poids sont rangés dans `%LOCALAPPDATA%/Snap2CAD/models`, `~/.local/share/Snap2CAD/models` ou `~/Library/Application Support/Snap2CAD/models` selon l'OS.
- Un `manifest.json` liste l'URL, la taille et le SHA-256 de chaque fichier.

### 3.2 Sécurité du worker local

Une page web ouverte dans un navigateur peut joindre `localhost`. Le worker doit donc :
- écouter **uniquement sur `127.0.0.1`**, sur un port aléatoire ;
- n'accepter que les requêtes portant un **jeton aléatoire** (en-tête `Authorization`), généré par Electron et transmis au lancement ;
- vérifier l'en-tête `Host` (protection contre le DNS rebinding) ;
- refuser le CORS ;
- n'exécuter le code CadQuery généré (mode Éditer) que dans un sous-processus isolé, avec délai maximal et sans accès réseau.

### 3.3 Abstraction du périphérique

Un module `snap2cad/device.py` centralise :
- le choix entre `cuda`, `mps` et `cpu` ;
- la précision : bf16 sur NVIDIA Ampere et plus récent, fp16 sur MPS (à valider), fp32 sur processeur ;
- la mémoire disponible : VRAM pour NVIDIA, mémoire unifiée pour les Mac ;
- le profil retenu : `full` (≥ 8 Go) ou `degraded` (6 Go).

Les étapes ne contiennent **aucun `torch.cuda` en dur**. Exemple : `vggt/demo_colmap.py` appelle `torch.cuda.get_device_capability` sans condition, il faut le réécrire.

---

## 4. Pipeline du mode Référence

> Schéma : [Annexe A2 — Pipeline du mode Référence](docs/annexes/ANNEXES.md#a2--pipeline-du-mode-référence-v2)

### 4.1 Différences avec le schéma initial (`Pipeline-photo-CAO.png`)

1. **VGGT tourne sur les photos entières, le détourage vient après.** Les objets visés sont peu texturés : ce sont la table et le décor qui donnent les poses.
2. **L'ajustement de faisceaux passe dans le tronc commun**, juste après VGGT. Toutes les étapes suivantes (échelle, vues, ICP) dépendent des poses.
3. **VGGT ne donne pas d'échelle réelle.** L'échelle vient de l'objet de référence, mesuré **dans les photos en pleine résolution** et non dans le nuage VGGT calculé à 518 px.
4. **Ajout d'une reconstruction dense en pleine résolution.** VGGT seul ne peut pas tenir ±0,5 mm.
5. **Ajout d'une étape de maillage** avant les exports.
6. **La complétion Hunyuan devient un repli**, après le guidage pour reprendre des photos.

### 4.2 Étapes

VRAM et temps sont des **estimations à mesurer au jalon M0**.

| # | Étape | Outil | Périph. | Entrée | Sortie |
|---|---|---|---|---|---|
| 1 | `import` | Pillow, pillow-heif, exifread | CPU | Dossier de photos | `images/` normalisées, `exif.json`, contrôles |
| 2 | `poses_init` | VGGT-1B, bf16 | GPU | Images entières (518 px) | `vggt/predictions.npz` : poses, intrinsèques, profondeur, confiance |
| 3 | `segment` | SAM 2.1, repli BiRefNet | GPU | Clics : objet, référence | `masks/object/*.png`, `masks/reference/*.png` |
| 4 | `ba` | pycolmap : ALIKED/SIFT en pleine résolution, ajustement de faisceaux | CPU (+GPU pour les points) | Images, poses VGGT, EXIF | `sparse/` (format COLMAP), `ba_report.json` |
| 5 | `scale` | OpenCV (ellipse et rectangle sous-pixel), triangulation, RANSAC du plan | CPU | Masques de référence, `sparse/` | `scale.json` : facteur, σ, repère monde |
| 6 | `dense` | COLMAP PatchMatch (CUDA) ou OpenMVS (Mac) | GPU / CPU | Images non distordues, masques objet | `dense/fused.ply` + confiance |
| 7 | `coverage` | Open3D (lancer de rayons) | CPU | Mesh grossier, caméras | `coverage.json`, `coverage.ply`, vues suggérées |
| 8 | `complete` *(repli)* | Hunyuan3D-2mv turbo + ICP avec échelle (Open3D) | GPU | 3 à 4 vues RGBA, nuage mesuré | Points `généré` fusionnés |
| 9 | `merge_sessions` *(option)* | FPFH + RANSAC + ICP, repli 3 points | CPU | Deux projets à l'échelle | Nuage fusionné, étiqueté par session |
| 10 | `mesh` | Open3D Poisson / pymeshlab, nettoyage | CPU | Nuage étiqueté | `mesh.ply` (attributs par sommet) |
| 11 | `uncertainty` | Modèle étalonné ([§4.5](#45-incertitude-par-sommet)) | CPU | Mesh, rapports des étapes | Attribut `sigma_mm` |
| 12 | `measure` | Ajustements point/cercle/plan, moindres carrés | CPU | Cotes saisies, sélections | `measurements.json`, recalage d'échelle |
| 13 | `export` | trimesh, OCP (STEP facetté) | CPU | Mesh final | STL, OBJ, PLY, GLB, STEP, `report.json` |

### 4.3 Échelle et repère (étape 5)

> Schéma : [Annexe A5 — Échelle et incertitude](docs/annexes/ANNEXES.md#a5--chaîne-déchelle-et-dincertitude)

**Pourquoi la pleine résolution :**
- L'erreur d'échelle relative vaut l'erreur sur le contour de la référence divisée par sa dimension.
- Sur une pièce de 25,75 mm, 0,1 mm d'erreur de contour donne 0,4 %, soit **1,2 mm sur 30 cm**.
- Il faut donc un contour précis **au sous-pixel**, obtenu sur plusieurs vues.

**Pièce de monnaie :**
1. Le masque SAM donne une zone de recherche.
2. Les contours sont détectés au sous-pixel (gradient + Devernay, ou contour SAM affiné), puis une ellipse est ajustée par vue (`cv2.fitEllipseDirect`, avec élimination des points aberrants).
3. Un cercle 3D (centre, normale, rayon) est ajusté pour reprojeter au mieux les ellipses de toutes les vues (≥ 5), avec les poses issues de l'ajustement de faisceaux.
4. Le cercle mesuré est la **face supérieure**, à 2,20 mm au-dessus de la table pour une pièce de 2 €. Ce n'est pas un problème : seul le diamètre sert à l'échelle.

**Carte ID-1 :**
1. On ajuste 4 droites (les bords) par vue, en écartant les coins arrondis (r = 3,18 mm).
2. On ajuste un rectangle 3D de 85,60 × 53,98 mm.
3. Les deux dimensions donnent deux estimations d'échelle, dont la cohérence est contrôlée.

**Repère :**
- Plan de table ajusté par RANSAC sur les points hors objet. Normale = +Z, origine = centre de la référence projeté sur la table, X aligné sur le grand axe de l'objet.

**Contrôles :**
- Écart entre vues.
- Accord des deux dimensions de la carte.
- Un σ d'échelle est produit et alimente l'incertitude.

### 4.4 Couverture et guidage (étape 7)

1. Construire un mesh grossier, puis échantillonner sa surface (environ 1 point/mm², adaptatif).
2. Pour chaque échantillon, lancer des rayons vers chaque caméra pour savoir lesquelles le voient réellement. Retenir celles dont l'angle d'incidence est inférieur à 75°, puis calculer **le plus grand angle de triangulation** entre ces caméras.
3. Un échantillon est couvert s'il est vu par au moins 3 caméras avec un angle d'au moins 15°.
4. Couverture = surface couverte / surface totale.
   - Si l'utilisateur a choisi « dessous non couvert », on **exclut la face en contact avec la table** (proche du plan, normale vers le bas).
5. En dessous de 90 % :
   - les zones manquantes s'affichent en rouge ;
   - **des positions de prise de vue sont suggérées** : on échantillonne des points de vue sur une sphère autour de l'objet et on retient ceux qui couvrent le plus de zones manquantes ;
   - la complétion n'est proposée qu'ensuite, clairement signalée.

### 4.5 Incertitude par sommet

Modèle additif (somme quadratique) :

```
σ_total² = σ_mvs² + σ_triang² + σ_scale² (+ σ_calib² si des cotes ont été saisies)
σ_triang ≈ k_t · σ_px · Z / (f · sin(θ_max))        (σ_px tiré de l'erreur de reprojection)
σ_mvs    = g(confiance VGGT/MVS, nombre de vues)       (régression étalonnée)
σ_scale  = (σ_s / s) · ‖p − origine‖                   (erreur d'échelle relative × distance)
σ_calib  = résidu des moindres carrés des cotes
```

- Les coefficients `k_t` et `g` sont **étalonnés sur les datasets métriques** ([§9](#9-stratégie-de-validation-et-datasets)).
- Objectif : environ 68 % des erreurs réelles à l'intérieur de 1σ.
- Les sommets `généré` reçoivent `sigma_mm = +∞` et sont exclus des outils de cote.

### 4.6 Cotes au pied à coulisse (étape 12)

- **Désignation :**
  - distance entre deux points cliqués ;
  - diamètre par cercle ajusté sur une sélection (RANSAC + moindres carrés) ;
  - épaisseur par distance entre deux plans ajustés.
- **Avec une seule cote :** affichage de l'écart et du σ prévu, sans correction.
- **À partir de deux :** facteur d'échelle par moindres carrés pondérés, affichage des résidus et ré-export.
- **Garde-fou :** si les résidus sont incohérents (par exemple > 3σ), on signale une erreur de saisie probable ou une déformation.

### 4.7 Complétion (étape 8, repli)

1. **Choix des vues :** dans le repère de la table, on prend les photos réelles les plus proches des azimuts 0°, 90°, 180° (et 270°), recadrées et détourées en RGBA. Elles alimentent les entrées `front`, `left`, `back` (et `right`) de Hunyuan3D-2mv.
2. **Génération :** Hunyuan3D-2mv turbo en fp16, 5 pas, FlashVDM. Le VAE turbo est téléchargé une fois, sinon on utilise `replace_vae=False`.
3. **Recalage :**
   - initialisation par boîte englobante et axes principaux, avec l'échelle du nuage mesuré ;
   - puis ICP avec échelle (`TransformationEstimationPointToPoint(with_scaling=True)`), **calculé uniquement sur les zones mesurées**.
4. **Fusion :** on ne garde du mesh généré que les zones à plus de τ de toute zone mesurée. Ces points reçoivent l'étiquette `généré`.
5. **Profil 6 Go :** désactivé par défaut, ou lancé avec `enable_model_cpu_offload` (lent) si l'utilisateur le demande.

### 4.8 Deux sessions (étape 9, option)

1. Chaque session est mise à l'échelle indépendamment, grâce à sa propre référence.
2. Alignement global FPFH + RANSAC, puis ICP point-à-plan.
3. **Détection d'ambiguïté :** si plusieurs solutions ont un score proche (objets à symétrie de révolution), l'utilisateur fait un alignement manuel par 3 paires de points.
4. Contrôle : écart d'échelle entre les deux sessions, signalé s'il dépasse 2σ.

---

## 5. Données, formats et dossier projet

> Schéma : [Annexe A4 — Dossier projet et cache](docs/annexes/ANNEXES.md#a4--dossier-projet-et-cache-par-hash)

### 5.1 Conventions

- **Unités :** mm partout après l'étape `scale`. Avant cette étape, unités VGGT (sans échelle réelle), indiquées dans `project.json`.
- **Repère monde :** table, Z vers le haut, main droite.
- **Caméras :** convention OpenCV (x à droite, y vers le bas, z vers l'avant), extrinsèques monde → caméra, comme VGGT et COLMAP.
- **Images :** RGB en 8 bits. L'orientation EXIF est appliquée à l'import, et l'original est conservé.

### 5.2 Formats d'échange

| Donnée | Format | Détails |
|---|---|---|
| Nuages, meshes | **PLY binaire** | Attributs par sommet : `source` (uint8 : 0 = mesuré, 1 = généré), `confidence` (float32), `n_views` (uint8), `tri_angle_deg` (float32), `sigma_mm` (float32), `session` (uint8) |
| Aperçu | **GLB** | Couleurs de sommets selon le mode d'affichage (texture, source, incertitude, couverture) |
| Caméras | COLMAP binaire (`cameras.bin`, `images.bin`, `points3D.bin`) | Plus `cameras.json` pour l'interface |
| Prédictions VGGT | `npz` | Poses, intrinsèques, profondeur, confiance |
| Masques | PNG 8 bits | Un dossier par cible (`object`, `reference`) |
| Rapports | JSON | `scale.json`, `coverage.json`, `measurements.json`, `report.json` |
| Exports | STL, OBJ, PLY, GLB, STEP facetté | En mm |

### 5.3 `project.json`

- Versionne chaque étape : paramètres, version du code, version et SHA-256 des modèles, hash des entrées, hash des sorties, statut, durée, pic de VRAM.
- Une étape est relancée **uniquement si le hash de ses entrées ou de ses paramètres a changé** (comme un outil de build, mais par fichier).
- Un exemple complet figure en [Annexe A4](docs/annexes/ANNEXES.md#a4--dossier-projet-et-cache-par-hash).

---

## 6. Environnement, dépendances et intégration des dépôts

### 6.1 Versions figées (à confirmer au jalon M0)

| Paquet | Version visée | Remarque |
|---|---|---|
| Python | 3.11 | |
| torch / torchvision | 2.7.x, CUDA 12.8 (Windows, Linux) ; version standard (Mac) | Pilote NVIDIA ≥ R570 requis |
| numpy | 1.26.4 | Contrainte VGGT (`numpy<2`) et pycolmap. CadQuery fonctionne aussi |
| transformers | 4.48.x | Hunyuan exige ≥ 4.48. CAD-Recode (prévu pour 4.47.1) est à tester |
| diffusers | Dernière version compatible torch 2.7 | Hunyuan |
| pycolmap | Version avec wheels sur les 3 OS | À vérifier pour Windows |
| open3d | 0.19.x | Wheels pour les 3 OS |
| opencv-contrib-python | 4.10.x | |
| sam2 | 2.1 | L'extension CUDA optionnelle est ignorée |
| cadquery / cadquery-ocp | Récente, avec wheels pour Windows et macOS arm64 | Phase 2 |
| fastapi, uvicorn, websockets | Récentes | |
| pillow-heif, exifread | Récentes | Photos HEIC des iPhone |

Hors Python :
- COLMAP en binaire CUDA (version Windows officielle ; à compiler ou empaqueter pour Linux) ;
- OpenMVS (Mac) ;
- Node et Electron pour l'interface.

### 6.2 Adaptations par dépôt

Commits figés : VGGT `a288dd0`, Hunyuan3D-2 `f8db630`, CAD-Recode `03e3262`.

**VGGT**
- Relâcher `torch==2.3.1` (test sur 2.7).
- Charger le modèle avec `VGGT.from_pretrained(<dossier>)`. `config.json` et `model.safetensors` sont présents, en fp32 (5,0 Go). Convertir **une fois en bf16** (environ 2,5 Go) dans le magasin de modèles.
- Rendre le choix du périphérique générique (`demo_colmap.py`, `projection.py:120` en float64 : passer sur le processeur pour MPS).
- **Embarquer** DINOv2 (`torch.hub`), le tracker VGGSfM, ALIKED et SuperPoint pour fonctionner hors ligne. Garder `keypoint_extractor="aliked+sp"` (usage personnel : la licence non commerciale de SuperPoint ne bloque pas).
- Ignorer `pyceres` (listé dans les dépendances mais jamais importé).

**Hunyuan3D-2**
- Charger avec `Hunyuan3DDiTFlowMatchingPipeline.from_single_file(model.fp16.safetensors, config.yaml)`, ou recréer l'arborescence `tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo/` sous `HY3DGEN_MODELS`.
- Ajouter `scikit-image` (importé mais absent des requirements).
- Ne pas installer `texgen` (rasteriseur C++/CUDA inutile).
- `mc_algo='mc'` (pas de `diso`). Pas de `torch.compile` ni de `sageattention` sous Windows.
- VAE turbo (`hunyuan3d-vae-v2-0-turbo`) : l'ajouter au manifeste des poids.

**CAD-Recode (phase 2)**
- Extraire la classe `CADRecode` de `demo.ipynb` dans un module.
- Remplacer `flash_attention_2` par `sdpa`, et pytorch3d (FPS) par `open3d` ou une version en torch.
- Embarquer le tokenizer `Qwen/Qwen2-1.5B`.
- Exécuter le code généré dans un sous-processus isolé.

### 6.3 Poids et téléchargements au premier lancement

| Groupe | Contenu | Taille approx. |
|---|---|---|
| Référence (obligatoire) | VGGT-1B bf16 (converti en local), SAM 2.1, BiRefNet, DINOv2-B (reg), tracker VGGSfM, ALIKED, SuperPoint | ≈ 5 à 6 Go (VGGT téléchargé en fp32 : 5 Go) |
| Complétion | Hunyuan3D-2mv turbo fp16 + VAE turbo | ≈ 5,5 Go |
| Éditer (phase 2) | CAD-Recode v1.5 + tokenizer (+ CADReasoner en option) | ≈ 3 à 7 Go |
| Environnement | Python 3.11, torch cu128, dépendances | ≈ 4 à 5 Go installé |

- Téléchargement reprenable (HTTP Range), vérification SHA-256, miroir Hugging Face.
- **Tout est téléchargé au premier lancement**, pour respecter le fonctionnement hors ligne.

---

## 7. Plateformes

> Matrice détaillée : [Annexe A8](docs/annexes/ANNEXES.md#a8--matrice-des-plateformes)

| | Windows (1ʳᵉ cible) | Ubuntu 22.04 / 24.04 | macOS Apple Silicon |
|---|---|---|---|
| Inférence | CUDA 12.8 | CUDA 12.8 | MPS (fp16) |
| Reconstruction dense | COLMAP PatchMatch (CUDA) | COLMAP CUDA (à compiler ou empaqueter) | OpenMVS (CPU) |
| Paquet | Installeur NSIS (electron-builder) | AppImage | `.dmg` non signé (`xattr -dr com.apple.quarantine`) |
| Points d'attention | SmartScreen sans signature | Pilote NVIDIA du système, base glibc | Ops float64 à passer sur le processeur, Hunyuan lent, `PYTORCH_ENABLE_MPS_FALLBACK=1` |

---

## 8. Gestion de la VRAM et modes dégradés

> Séquence de chargement et déchargement : [Annexe A3](docs/annexes/ANNEXES.md#a3--séquence-dexécution-et-vram)

**Règles :**
- une étape GPU = un sous-processus ;
- un seul modèle en VRAM ;
- la VRAM de chaque processus est plafonnée à *libre au lancement − 256 Mo* (`device.cap_cuda_memory`) ;
- le pic de VRAM est mesuré (`torch.cuda.max_memory_allocated`) et enregistré dans `project.json`.

| Étape | Profil `full` (≥ 8 Go) | Profil `degraded` (6 Go) | Estimation VRAM (à mesurer) |
|---|---|---|---|
| VGGT | bf16, jusqu'à N₈ images | bf16, N₆ images (traitement par lots + ajustement global si dépassement) | 2,5 Go de poids + activations |
| SAM 2.1 | `large` | `base_plus` | 1 à 2 Go |
| Points pour l'ajustement | aliked+sp, 4096 points | 2048 points | < 2 Go |
| Reconstruction dense COLMAP | `max_image_size` 3200 | 2000 | 2 à 4 Go |
| Hunyuan3D-2mv | fp16, FlashVDM | **désactivé** (ou déchargement processeur sur demande) | ≈ 6 Go |
| CAD-Recode | bf16, sdpa | bf16, sdpa | ≈ 3,5 Go |

N₈ et N₆ sont fixés au jalon M0 par mesure sur la RTX 4060 et la RTX 3060 Laptop ([docs/benchmarks/M0-vggt.md](docs/benchmarks/M0-vggt.md)). Chaque processus plafonne sa VRAM à *libre au lancement − 256 Mo* (`device.cap_cuda_memory`) : sans ce plafond, Windows déborde en mémoire partagée au lieu d'échouer, et les étapes deviennent 2 à 3 fois plus lentes. **Mesuré : N₈ = 28** pour des photos 4:3, sous ce plafond (36 avec un bureau léger). N₆ = 20 avait été mesuré avant le plafond : il reste à remesurer sur la 3060.

---

## 9. Stratégie de validation et datasets

> Correspondance datasets / jalons : [Annexe A7](docs/annexes/ANNEXES.md#a7--plan-de-validation)

### 9.1 Principe : séparer l'erreur de forme et l'erreur d'échelle

Aucun dataset réel ne contient de pièce de 2 € ni de carte bancaire. L'erreur totale est donc décomposée :
- **erreur de forme :** reconstruction comparée à la vérité terrain **après alignement optimal (rotation, translation, échelle)**. Mesurée sur les datasets réels ;
- **erreur d'échelle :** facteur estimé comparé au facteur exact. Mesurée sur le **jeu synthétique**, puis sur tes propres photos.

### 9.2 Datasets

| Dataset | Rôle | Statut |
|---|---|---|
| **DTU MVS 2014 – SampleSet** (scènes 1 et 6, code d'évaluation officiel) | Précision et complétude de la reconstruction dense, poses | **Téléchargé** → `datasets/dtu/` |
| **MobileBrick** (LEGO filmés à l'iPhone, vérité terrain exacte, 18 séquences d'évaluation) | Erreur de forme en conditions smartphone, F1 à 2,5 et 5 mm (plus 0,5 et 1 mm ajoutés par nous) | **Téléchargé** → `datasets/mobilebrick/` |
| T-LESS (pièces sans texture, modèles CAO) | Cas peu texturé : trous, ajustement de primitives | À télécharger (licence à vérifier) |
| Digital Twin Catalog (objets du quotidien, précision de l'ordre du mm) | Robustesse, objets noirs, étalonnage de l'incertitude | **À télécharger par l'utilisateur** (acceptation de licence requise) |
| Stanford-ORB (14 objets scannés) | Validation secondaire | Optionnel |
| CO3D / uCO3D (vidéos smartphone sans échelle) | Robustesse du détourage et de la couverture | Optionnel |
| **Jeu synthétique Snap2CAD** (Google Scanned Objects CC-BY 4.0 sous Blender) | Échelle (pièce, carte), couverture, deux sessions, noir mat, vérité terrain exacte | **À construire (jalon M1b)** |

### 9.3 Jeu synthétique (outil `tools/synth/`)

- **Objets :** environ 30 objets Google Scanned Objects de 5 à 30 cm. Matière d'origine, plus une variante forcée en **noir mat** (albédo ≈ 0,04, rugosité 0,7).
- **Références :** pièce de 2 € (Ø 25,75 × 2,20 mm, tranche striée), pièce de 1 € (Ø 23,25 mm), carte ID-1 (85,60 × 53,98 mm, coins r = 3,18 mm).
- **Tables :** unie, bois, motif.
- **Caméras « smartphone » :** focale équivalente 24 à 26 mm, 4032 × 3024, légère distorsion radiale, bruit, flou de bougé, compression JPEG ou HEIC. 15 à 40 vues à 2 ou 3 hauteurs.
- **Variantes :**
  - séries incomplètes (une zone non photographiée) pour tester la couverture ;
  - séries « objet retourné » pour tester la fusion ;
  - quelques séries sans référence visible sur au moins 5 vues, pour tester les erreurs.
- **Vérité terrain :** mesh, poses, intrinsèques, masques, profondeur, facteur d'échelle, cotes caractéristiques (hors-tout, diamètres, entraxes) extraites automatiquement ou annotées.
- **Outil :** Blender (Cycles) piloté par script Python. BlenderProc (GPL-3.0) est possible car il sert uniquement d'outil, sans être distribué avec l'app.

### 9.4 Métriques et critères d'acceptation (provisoires)

| Métrique | Calcul | Cible provisoire |
|---|---|---|
| Erreur de rotation des poses | Médiane après alignement avec échelle | < 1° (DTU, MobileBrick) |
| Erreur de forme | Distance de chanfrein ; précision et complétude (mm) ; F-score à 0,5 / 1 / 2,5 mm | DTU : comparable ou meilleure que notre propre référence COLMAP seule. MobileBrick : F1 à 2,5 mm ≥ référence publiée |
| Erreur d'échelle relative | \|s_est / s_vrai − 1\| | Pièce : < 0,15 %. Carte : < 0,08 % (soit environ 0,45 et 0,25 mm sur 30 cm) |
| Erreur de cote | Cotes caractéristiques (synthétique) | P90 ≤ 0,5 mm ; ≤ 0,2 mm après recalage par 2 cotes |
| Étalonnage de l'incertitude | Part des erreurs ≤ 1σ | 68 % ± 10 % |
| Couverture | Zones manquantes détectées dans les séries incomplètes | Rappel ≥ 95 % |

**Validation finale réelle (obligatoire avant de considérer M5 atteint) :** 3 à 5 objets réels du périmètre, dont au moins un noir mat, photographiés au smartphone avec une pièce et une carte, et 5 à 10 cotes au pied à coulisse par objet. À faire dès ton retour.

---

## 10. Jalons du mode Référence

> Graphe des dépendances : [Annexe A6](docs/annexes/ANNEXES.md#a6--jalons-et-dépendances)
> Taille : S ≈ quelques jours, M ≈ 1 à 2 semaines, L ≈ 2 à 4 semaines (ordre de grandeur pour un développeur, **hypothèse**).

### M0 — Socle technique · M
**Objectif :** un environnement reproductible et des mesures de référence sur les deux GPU.

**Tâches :**
- Arborescence du dépôt ([§13](#13-organisation-du-dépôt-et-conventions)), dépôts tiers dans `third_party/` à commits figés.
- Environnement Python 3.11 + torch 2.7 cu128 (`uv`, fichier `uv.lock`), script `tools/check_env.py` (GPU, pilote, CUDA, versions).
- `snap2cad/device.py` (profils `full` et `degraded`), `snap2cad/project/` (`project.json`, hash, cache).
- Conversion de VGGT en bf16, magasin de modèles et `manifest.json`.
- VGGT sur `vggt/examples/kitchen` et sur DTU scène 1 : mesure de VRAM et de temps selon le nombre d'images, sur la 3060 et la 4060.

**Critères :**
- `check_env` est vert sur les deux machines.
- Tableau VRAM et temps rempli.
- N₈ et N₆ fixés.

### M1 — Import et poses initiales · M
**Tâches :**
- Étape `import` : JPEG et HEIC, orientation EXIF, focale équivalente 35 mm → intrinsèques initiaux.
- Contrôle de l'objectif via `LensModel` : refus ou avertissement en cas d'ultra grand-angle, y compris la **bascule automatique en macro de l'iPhone**.
- Détection de flou (variance du laplacien), contrôle du nombre de photos.
- Étape `poses_init` (VGGT) : `npz`, PLY coloré par confiance, GLB.
- Adaptateurs de lecture DTU et MobileBrick (poses, intrinsèques et vérité terrain vers notre format).

**Critères :** sur DTU et MobileBrick, la médiane d'erreur de rotation est mesurée et enregistrée (référence pour M3).

### M1b — Jeu synthétique · M (en parallèle de M1–M2)
**Tâches :** générateur Blender décrit au [§9.3](#93-jeu-synthétique-outil-toolssynth). Premier lot : 10 objets × 3 variantes.

**Critères :**
- La vérité terrain se relit avec nos adaptateurs.
- Une série reconstruite par VGGT s'aligne sur la vérité terrain.

### M2 — Détourage · M
**Tâches :**
- SAM 2.1 en mode vidéo, photos triées par heure EXIF, clic sur la 1ʳᵉ photo.
- **Repli guidé en 3D :** le clic est remonté en 3D avec la profondeur VGGT, puis reprojeté dans chaque vue comme invite (points ou boîte) pour SAM en mode image.
- Deux cibles : `object` et `reference` (choix dans la liste pièce 2 € / 1 € / carte).
- BiRefNet en repli automatique.
- Correction manuelle par clics positifs et négatifs sur une vue.

**Critères :** IoU des masques ≥ 0,9 sur le synthétique, et aucune perte de l'objet sur les séries de 20 vues espacées.

### M3 — Ajustement de faisceaux en pleine résolution · M
**Tâches :**
- Points (ALIKED+SP) en pleine résolution, suivis initialisés par VGGT.
- Ajustement pycolmap avec caméra partagée par objectif, modèle `SIMPLE_RADIAL` ou `RADIAL`, intrinsèques initiaux tirés de l'EXIF.
- Filtrage par erreur de reprojection. Rapport `ba_report.json`.

**Critères :**
- Erreur de reprojection médiane < 1 px.
- Erreur de rotation inférieure à celle de VGGT seul (M1) sur DTU et MobileBrick.

### M4 — Échelle et repère · M · **point de décision**
**Tâches :**
- Contour sous-pixel et ellipse (pièce) ou rectangle (carte), cercle ou rectangle 3D sur plusieurs vues ([§4.3](#43-échelle-et-repère-étape-5)).
- Plan de table et repère monde.
- σ d'échelle.

**Critères (synthétique) :** erreur d'échelle relative < 0,15 % (pièce) et < 0,08 % (carte).

**Décision :** si la cible n'est pas atteinte → **carte obligatoire et cotes au pied à coulisse obligatoires** (décision déjà validée). Il reste à revalider sur photos réelles dès que possible.

### M5 — Reconstruction dense, maillage, exports · L
**Tâches :**
- Images non distordues, PatchMatch COLMAP (CUDA) limité aux masques objet, fusion.
- Ajustement de primitives (plans, cylindres) pour combler les trous des grandes faces noires uniformes, **étiquetés à part** (`source=mesuré-ajusté`, sous-type de mesuré, σ dédié).
- Poisson ou ball-pivoting, nettoyage (composantes, fermeture optionnelle).
- Exports STL, OBJ, PLY, GLB et STEP facetté (OCP : couture des triangles).

**Critères :**
- Précision et complétude DTU mesurées.
- F-score MobileBrick à 0,5, 1 et 2,5 mm.
- Erreur de cote P90 ≤ 0,5 mm sur le synthétique.
- Les exports s'ouvrent dans FreeCAD et Fusion 360, à la bonne échelle.
- **Validation réelle :** une fois tes photos disponibles.

### M6 — Couverture et guidage · M
**Tâches :** [§4.4](#44-couverture-et-guidage-étape-7). Sortie `coverage.ply`, carte des zones manquantes, vues suggérées. Ajout incrémental de photos : l'orchestrateur ne relance que les étapes concernées.

**Critères :** rappel ≥ 95 % sur les séries incomplètes du synthétique.

### M7 — Incertitude et cotes au pied à coulisse · M
**Tâches :**
- Modèle d'incertitude ([§4.5](#45-incertitude-par-sommet)), étalonné sur DTU, MobileBrick et le synthétique.
- Étape `measure` : point-point, cercle, plan-plan, recalage à partir de 2 cotes, résidus.

**Critères :**
- 68 % ± 10 % des erreurs à l'intérieur de 1σ.
- Cotes recalées P90 ≤ 0,2 mm sur le synthétique.

### M8 — Application Electron (1ʳᵉ version utilisable) · L
**Tâches :**
- Worker FastAPI + WebSocket (sécurité du [§3.2](#32-sécurité-du-worker-local)), orchestrateur avec annulation et reprise.
- Interface React + three.js :
  - gestion des projets, import ;
  - clics de détourage ;
  - vue 3D (modes texture, source, incertitude, couverture) ;
  - outils de mesure ;
  - assistant de prise de vue (rappels : référence visible sur ≥ 5 photos, objectif principal, éclairage diffus, support texturé si l'objet est peu texturé, spray optionnel) ;
  - exports et rapport.
- Test de charge de l'affichage : 5 à 10 millions de points. Bascule vers PySide6 seulement en cas d'échec.

**Critères :** démo de bout en bout sur une série synthétique et une série MobileBrick, sans ligne de commande.

### M9 — Deux sessions et fusion · M
**Tâches :** [§4.8](#48-deux-sessions-étape-9-option). Interface d'alignement par 3 points.

**Critères :**
- Fusion réussie sur les séries « retournées » du synthétique.
- Ambiguïté détectée sur les objets de révolution.

### M10 — Complétion (repli) · M
**Tâches :** [§4.7](#47-complétion-étape-8-repli). Étiquetage `généré`, blocage des cotes sur ces zones, avertissement dans le rapport.

**Critères :**
- Aucun sommet `généré` n'est utilisable pour une cote.
- Fonctionne en profil `full`, désactivé en `degraded`.

### M11 — Distribution · L
**Tâches :**
- Installeur Windows (NSIS), premier lancement :
  - environnement créé par `uv` ;
  - téléchargement reprenable des poids ;
  - vérification SHA-256 ;
  - test hors ligne.
- Puis AppImage Linux, puis `.dmg` Mac (portage MPS, OpenMVS, mesures de performance).

**Critères :** installation sur une machine vierge pour chaque OS, puis un projet complet en hors ligne.

> **Chemin critique :** M0 → M1 → M3 → M4 → M5 → M7 → M8 → M11.
> M1b, M2 et M6 peuvent avancer en parallèle. M9 et M10 viennent après M8.

---

## 11. Mode Éditer (phase 2)

| Phase | Contenu | Critère |
|---|---|---|
| E1 | Prototype CAD-Recode v1.5 : nuage à l'échelle → normalisation `[-1,1]` → FPS 256 points → génération (sdpa) → exécution isolée (délai maximal) → retour aux mm → STEP + `.py` | Jeu de test CAD-Recode : erreur de chanfrein du même ordre que le papier. Objets synthétiques simples réussis |
| E2 | Comparaison avec CADReasoner-PC (Qwen2-VL-2B, 128 points, 3 itérations × 4 échantillons, Apache-2.0 selon la fiche du modèle) : flash-attn → sdpa, VRAM et temps sur 6 et 8 Go | Choix du modèle par défaut |
| E3 | Débruitage et normalisation spécifiques (zones `généré` exclues ou pondérées), découpage en pièces rigides si nécessaire | — |
| E4 | Interface d'édition : script CadQuery modifiable avec aperçu en direct (ré-exécution isolée), comparaison avec le nuage mesuré | — |

---

## 12. Risques et parades

| # | Risque | Impact | Parade |
|---|---|---|---|
| R1 | Surfaces noir mat peu texturées : trous et bruit en reconstruction dense | ±0,5 mm non tenu | Support texturé, ajustement de primitives, spray optionnel, contour par silhouettes multiples (à étudier) |
| R2 | Échelle par pièce insuffisamment précise | Erreur proportionnelle à la taille | Contour sous-pixel sur plusieurs vues, carte au-delà de 15 cm, recalage par cotes (déjà décidé) |
| R3 | Propagation SAM 2.1 perdue sur des vues très espacées | Masques faux | Repli guidé en 3D, correction par clic, BiRefNet |
| R4 | Datasets publics peu représentatifs (aucun avec pièce, rarement smartphone + noir) | Critères optimistes | Synthétique noir mat, validation réelle obligatoire avant de clore M5 |
| R5 | VRAM de 6 Go insuffisante pour VGGT avec beaucoup d'images | Échec ou lenteur | Lots + ajustement global, moins d'images, mesures en M0 |
| R6 | torch 2.7 non compatible avec VGGT (prévu pour 2.3.1) ou CAD-Recode (transformers) | Blocage d'environnement | Tests en M0. Repli sur deux environnements |
| R7 | COLMAP CUDA sous Linux (pas de binaire officiel) | Retard M11 | Compilation dans l'intégration continue ou repli OpenMVS |
| R8 | MPS : float64 non supporté, opérations manquantes | Retard du portage Mac | Passage sur processeur ciblé, `PYTORCH_ENABLE_MPS_FALLBACK` |
| R9 | Bascule macro automatique de l'iPhone, photos HEIC, traitements du téléphone | Intrinsèques faux | Contrôle `LensModel`, intrinsèques affinés par l'ajustement de faisceaux, consigne « objectif 1× » |
| R10 | Licences : VGGT-1B et CAD-Recode non commerciaux, Hunyuan avec restriction de territoire | Bloque toute diffusion publique | Usage personnel uniquement. À réévaluer si le projet change de statut |
| R11 | Environ 15 à 18 Go à télécharger au premier lancement | Mauvaise expérience | Reprise, vérification, progression détaillée, groupes de poids |
| R12 | Affichage de gros nuages dans Electron | Lenteur de l'interface | Réduction du nombre de points pour l'affichage, chargement progressif, `three-mesh-bvh`. Repli PySide6 |

---

## 13. Organisation du dépôt et conventions

```
Snap2CAD/
├── PLAN_DE_DEVELOPPEMENT.md
├── Pipeline-photo-CAO.png
├── app/                       # Electron + React + three.js (TypeScript)
├── worker/
│   ├── pyproject.toml         # paquet snap2cad, géré par uv (uv.lock)
│   ├── snap2cad/
│   │   ├── api/               # FastAPI + WebSocket
│   │   ├── orchestrator/      # graphe d'étapes, cache, sous-processus
│   │   ├── steps/             # import, poses_init, segment, ba, scale, dense, coverage,
│   │   │                      # complete, merge_sessions, mesh, uncertainty, measure, export
│   │   ├── models/            # adaptateurs VGGT, SAM2, BiRefNet, Hunyuan, CAD-Recode
│   │   ├── geometry/          # ajustements, ICP, primitives, lancer de rayons
│   │   ├── project/           # project.json, hash, formats PLY/GLB
│   │   └── device.py
│   └── tests/                 # pytest : unitaires + tests sur données de référence
├── third_party/               # vggt, Hunyuan3D-2, cad-recode (commits figés, patchs documentés)
├── tools/
│   ├── synth/                 # générateur Blender du jeu synthétique
│   ├── eval/                  # adaptateurs DTU / MobileBrick, métriques
│   └── check_env.py
├── models/                    # (dev) poids + manifest.json — hors git
├── datasets/                  # DTU, MobileBrick, synthétique — hors git
└── docs/
    └── annexes/               # visuels du plan
```

**Conventions :**
- Code et identifiants en anglais, documentation utilisateur en français (à confirmer, voir [§14](#14-questions-encore-ouvertes)).
- Python : `ruff` et `mypy` (mode strict progressif), `pytest`. TypeScript : `eslint` et `prettier`.
- **Intégration continue (GitHub Actions) :** lint et tests processeur sur Windows, Ubuntu et macOS. Les tests GPU se lancent manuellement (`pytest -m gpu`) sur la 3060 et la 4060.
- **Tests de non-régression de précision :** un petit sous-ensemble (1 scène DTU, 2 séquences MobileBrick, 5 séries synthétiques) avec des seuils enregistrés. Une baisse de précision fait échouer les tests.
- Journaux structurés (JSON) par étape, conservés dans le dossier projet.

---

## 14. Questions encore ouvertes

Aucune ne bloque le démarrage de M0.

1. Hébergement du dépôt et intégration continue : GitHub (privé) ?
2. Langue du code et des commentaires : anglais (proposé) ou français ?
3. ~~D'accord pour déplacer `vggt/`, `Hunyuan3D-2/` et `cad-recode/` dans `third_party/`, et les poids dans `models/`, au jalon M0 ?~~ Fait en M0 (sous-modules git aux commits figés).
4. Blender est-il déjà installé ? Il faut une version 4.x pour le jeu synthétique.
5. Rythme visé (heures par semaine) pour transformer les tailles S/M/L en calendrier ?

---

*Annexes visuelles : [docs/annexes/ANNEXES.md](docs/annexes/ANNEXES.md)*
