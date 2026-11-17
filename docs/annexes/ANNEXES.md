# Annexes du plan de développement

Retour au plan : [PLAN_DE_DEVELOPPEMENT.md](../../PLAN_DE_DEVELOPPEMENT.md)

- **A1 et A2** sont des images SVG.
- **A3 à A9** sont des diagrammes Mermaid. Ils s'affichent sur GitHub et GitLab, dans Obsidian, et dans VS Code avec l'extension *Markdown Preview Mermaid Support*.

---

## A1 — Architecture globale

![Architecture globale](A1-architecture.svg)

- **Electron** ne fait aucun calcul : il affiche les fichiers produits par les étapes et envoie des commandes au worker.
- **Le worker** orchestre les étapes. Chaque étape tourne dans un sous-processus qui libère toute sa mémoire en se terminant.
- **Le dossier projet** est la seule source de vérité. On peut le copier, l'archiver ou le rejouer en ligne de commande.

---

## A2 — Pipeline du mode Référence (v2)

![Pipeline du mode Référence](A2-pipeline-reference.svg)

Différences avec le schéma initial (`Pipeline-photo-CAO.png`) :
- VGGT tourne sur les photos entières, le détourage vient après ;
- ajustement de faisceaux et échelle dans le tronc commun ;
- reconstruction dense et maillage ajoutés ;
- complétion reléguée au rang de repli, après le guidage pour reprendre des photos.

---

## A3 — Séquence d'exécution et VRAM

Une seule étape GPU à la fois. La mémoire est libérée par la fin du sous-processus, pas par un simple `del model`.

```mermaid
sequenceDiagram
    autonumber
    participant UI as Interface (Electron)
    participant W as Worker / Orchestrateur
    participant P as Sous-processus d'étape
    participant G as GPU (VRAM)
    participant D as Dossier projet

    UI->>W: POST /projects/{id}/run (cible : export)
    W->>D: lit project.json, calcule les hash
    Note over W: Plan : import → poses_init → segment → ba → scale → dense → coverage → mesh → uncertainty → export<br/>(les étapes dont le hash n'a pas changé sont sautées)

    W->>P: lance poses_init (VGGT)
    P->>G: charge VGGT bf16 (≈ 2,5 Go + activations)
    P->>D: écrit vggt/predictions.npz
    P-->>W: fin (code 0, pic de VRAM)
    Note over G: processus terminé → VRAM libérée

    W->>P: lance segment (SAM 2.1)
    P->>G: charge SAM 2.1 (large ou base+ selon le profil)
    UI->>W: clics (objet, référence)
    W->>P: transmet les invites
    P->>D: écrit masks/
    P-->>W: fin

    W->>P: lance ba, puis scale (CPU)
    P->>D: écrit sparse/, scale.json

    W->>P: lance dense (COLMAP CUDA / OpenMVS)
    P->>G: PatchMatch (taille d'image selon le profil)
    P->>D: écrit dense/fused.ply
    P-->>W: fin

    W->>P: lance coverage (CPU)
    P->>D: écrit coverage.json
    alt couverture < 90 %
        W-->>UI: zones manquantes + vues suggérées
        UI->>W: photos ajoutées → relance incrémentale
    else couverture ≥ 90 % (ou dessous accepté non couvert)
        W->>P: mesh → uncertainty → export (CPU)
        P->>D: mesh.ply, preview.glb, exports/, report.json
    end
    W-->>UI: WebSocket : terminé, URL du GLB
```

---

## A4 — Dossier projet et cache par hash

### Arborescence

```
mon_objet.s2c/
├── project.json            # état, paramètres, hash, versions (voir ci-dessous)
├── input/                  # photos d'origine (inchangées)
├── images/                 # photos normalisées (orientation EXIF appliquée)
├── exif.json
├── vggt/predictions.npz    # poses, intrinsèques, profondeur, confiance (sans échelle)
├── masks/
│   ├── object/*.png
│   └── reference/*.png
├── sparse/                 # COLMAP : cameras.bin, images.bin, points3D.bin
├── ba_report.json
├── scale.json              # facteur, σ, type de référence, repère monde
├── dense/fused.ply
├── coverage.json  coverage.ply
├── completion/             # (repli) mesh Hunyuan brut + transformation ICP
├── mesh.ply                # attributs : source, confidence, n_views, tri_angle_deg, sigma_mm, session
├── preview.glb
├── measurements.json
├── exports/                # STL, OBJ, PLY, GLB, STEP
├── report.json
└── logs/<étape>.jsonl
```

### Exemple de `project.json`

```json
{
  "schema_version": 1,
  "app_version": "0.1.0",
  "mode": "reference",
  "units": "mm",
  "world_frame": "table_z_up",
  "reference_object": { "type": "coin_2eur", "diameter_mm": 25.75, "thickness_mm": 2.20 },
  "options": { "underside": "not_covered", "profile": "full", "device": "cuda:0" },
  "sessions": [ { "id": 0, "input_dir": "input/", "n_images": 28 } ],
  "steps": {
    "poses_init": {
      "status": "done",
      "params": { "model": "vggt-1b", "dtype": "bf16", "resolution": 518 },
      "model_sha256": "…",
      "code_version": "snap2cad 0.1.0 + vggt@a288dd0",
      "inputs_hash": "sha256:…",
      "outputs": { "vggt/predictions.npz": "sha256:…" },
      "duration_s": 41.2,
      "peak_vram_mb": 5310
    },
    "scale": {
      "status": "done",
      "params": { "method": "ellipse_multiview", "min_views": 5 },
      "inputs_hash": "sha256:…",
      "outputs": { "scale.json": "sha256:…" },
      "result": { "scale": 87.31, "sigma_rel": 0.0011, "n_views": 9 }
    },
    "dense": { "status": "stale", "reason": "inputs_hash changed (masks/object)" }
  }
}
```

### Décision de relance

```mermaid
flowchart LR
    A[Étape demandée] --> B{Sorties présentes<br/>et hash valides ?}
    B -- non --> R[Relancer l'étape]
    B -- oui --> C{hash des entrées<br/>+ paramètres<br/>+ version code/modèle<br/>inchangés ?}
    C -- oui --> S[Sauter : réutiliser le cache]
    C -- non --> R
    R --> I[Marquer les étapes en aval « stale »]
    I --> N[Étape suivante]
    S --> N
```

---

## A5 — Chaîne d'échelle et d'incertitude

```mermaid
flowchart TB
    subgraph REF["Objet de référence (choisi dans une liste)"]
        R1[Pièce 2 € · Ø 25,75 mm · ép. 2,20 mm]
        R2[Pièce 1 € · Ø 23,25 mm]
        R3[Carte ID-1 · 85,60 × 53,98 mm<br/>obligatoire · décision M4]
    end

    REF --> K[Clic sur une photo → SAM 2.1<br/>masque sur toutes les vues]
    K --> E1[Contours sous-pixel<br/>en pleine résolution]
    E1 --> E2{Type ?}
    E2 -- pièce --> P1[Ellipse par vue<br/>fitEllipseDirect + rejet des aberrants]
    E2 -- carte --> P2[4 droites par vue<br/>coins arrondis exclus]
    P1 --> T[Cercle 3D ajusté sur ≥ 5 vues<br/>avec les poses de l'ajustement de faisceaux]
    P2 --> T2[Rectangle 3D ajusté<br/>2 dimensions → contrôle de cohérence]
    T --> F[Facteur d'échelle s + σ_s]
    T2 --> F
    F --> PL[Plan de table RANSAC → Z vers le haut<br/>origine = centre de la référence]
    PL --> MM[Toute la géométrie en mm]

    MM --> U
    subgraph U["Incertitude par sommet (σ en mm)"]
        U1[σ_mvs : confiance VGGT/MVS + nb de vues]
        U2[σ_triang : erreur de reprojection,<br/>profondeur, angle max]
        U3[σ_scale : σ_s/s × distance à l'origine]
        U4[σ_calib : résidus des cotes au pied à coulisse]
    end
    U --> SUM["σ_total = √(Σ σ²)<br/>étalonnée : 68 % des erreurs ≤ 1σ"]

    C[Cotes au pied à coulisse] --> C1{Combien ?}
    C1 -- 1 --> V[Vérification : écart vs σ prévu]
    C1 -- "≥ 2" --> LS[Moindres carrés pondérés<br/>→ nouveau facteur + résidus]
    LS --> MM
    LS --> U4
```

---

## A6 — Jalons et dépendances

Le chemin critique est en gras (bordure orange).

```mermaid
flowchart LR
    M0["M0 Socle<br/>env, device, project.json<br/>mesures VRAM 3060/4060"]:::crit
    M1["M1 Import + VGGT<br/>EXIF, HEIC, objectif"]:::crit
    M1b["M1b Jeu synthétique<br/>Blender + GSO"]
    M2["M2 Détourage<br/>SAM 2.1 + repli 3D"]
    M3["M3 Ajustement de faisceaux<br/>pleine résolution"]:::crit
    M4{{"M4 Échelle + repère<br/>POINT DE DÉCISION"}}:::crit
    M5["M5 Reconstruction dense<br/>+ maillage + exports"]:::crit
    M6["M6 Couverture<br/>+ guidage"]
    M7["M7 Incertitude<br/>+ cotes au pied à coulisse"]:::crit
    M8["M8 App Electron<br/>1re version utilisable"]:::crit
    M9["M9 Deux sessions<br/>+ fusion"]
    M10["M10 Complétion<br/>Hunyuan (repli)"]
    M11["M11 Distribution<br/>Win → Linux → Mac"]:::crit
    RV(["Validation réelle<br/>tes photos + pied à coulisse"]):::real

    M0 --> M1 --> M3 --> M4 --> M5 --> M7 --> M8 --> M11
    M0 --> M1b
    M1 --> M2
    M1b --> M2
    M2 --> M4
    M1b --> M4
    M5 --> M6
    M6 --> M8
    M8 --> M9
    M8 --> M10
    M4 -. "échec → carte + cotes obligatoires" .-> M7
    RV -.-> M5

    classDef crit stroke:#f0a35e,stroke-width:3px
    classDef real fill:#b9e3c6,color:#1c1c1e
```

| Jalon | Taille (hypothèse) | Sortie vérifiable |
|---|---|---|
| M0 | M | `check_env` vert, tableau VRAM et temps, N₈ et N₆ fixés |
| M1 | M | Poses VGGT, erreur de rotation sur DTU et MobileBrick |
| M1b | M | 10 objets × 3 variantes avec vérité terrain |
| M2 | M | IoU des masques ≥ 0,9 (synthétique) |
| M3 | M | Reprojection < 1 px, rotation meilleure que M1 |
| M4 | M | Erreur d'échelle < 0,15 % (pièce) et < 0,08 % (carte) |
| M5 | L | Cotes P90 ≤ 0,5 mm (synthétique), F-score MobileBrick |
| M6 | M | Rappel des zones manquantes ≥ 95 % |
| M7 | M | 68 % ± 10 % dans 1σ ; cotes recalées ≤ 0,2 mm |
| M8 | L | Démo de bout en bout sans ligne de commande |
| M9 | M | Fusion des séries retournées, ambiguïté détectée |
| M10 | M | Aucun sommet « généré » utilisable pour une cote |
| M11 | L | Installation vierge et projet hors ligne sur chaque OS |

---

## A7 — Plan de validation

```mermaid
flowchart LR
    subgraph REAL["Datasets réels (erreur de forme)"]
        DTU["DTU SampleSet<br/>scènes 1 et 6<br/>scans par lumière structurée"]
        MB["MobileBrick test<br/>18 séquences iPhone<br/>vérité terrain LEGO exacte"]
        TL["T-LESS<br/>pièces sans texture + CAO"]
        DTC["Digital Twin Catalog<br/>objets du quotidien<br/>(licence à accepter)"]
    end
    subgraph SYN["Jeu synthétique (vérité terrain exacte)"]
        GSO["GSO + Blender<br/>pièce / carte, noir mat,<br/>séries incomplètes et retournées"]
    end
    subgraph USER["Validation finale"]
        RV["Tes objets réels<br/>smartphone + pièce + carte<br/>+ cotes au pied à coulisse"]
    end

    DTU --> Mp["Poses : rotation médiane"]
    MB --> Mp
    DTU --> Mf["Forme : précision / complétude,<br/>F-score à 0,5 / 1 / 2,5 mm"]
    MB --> Mf
    TL --> Mf
    DTC --> Mf
    GSO --> Ms["Échelle : erreur relative"]
    GSO --> Mc["Cotes : P90 en mm"]
    GSO --> Mv["Couverture : rappel"]
    GSO --> Mm["Masques : IoU"]
    DTU --> Mu["Incertitude : étalonnage"]
    MB --> Mu
    GSO --> Mu
    RV --> Mall["Erreur totale réelle<br/>(critère final M5 / M7)"]

    Mp --> J1[M1, M3]
    Mf --> J5[M5]
    Ms --> J4[M4]
    Mc --> J5
    Mv --> J6[M6]
    Mm --> J2[M2]
    Mu --> J7[M7]
    Mall --> J5
```

**État des téléchargements (11/11/2026) :**

| Dataset | Archive | Taille | Extrait dans |
|---|---|---|---|
| DTU MVS 2014 SampleSet | `SampleSet.zip` (roboimagedata2.compute.dtu.dk) | 6,9 Go | `datasets/dtu/` |
| MobileBrick (split test uniquement, 18 séquences) | `MobileBrick_Mar23.zip` (robots.ox.ac.uk) | 13,1 Go (test : 2,0 Go) | `datasets/mobilebrick/test/` |

Contenu vérifié :
- **DTU :** scènes 1 et 6 avec images `Cleaned` et `Rectified` (49 positions × 7 éclairages), calibration, nuages de référence `Points/stl/stl00{1,6}_total.ply`, masques d'évaluation `ObsMask` et plans, résultats des méthodes de référence (camp, furu, tola), code d'évaluation MATLAB.
- **MobileBrick :** 18 séquences, 9 868 fichiers. Images 1920 × 1440, intrinsèques et poses par image (caméra → monde, **en mètres**), masques, profondeur de référence, `mesh/gt_mesh.ply`.

L'archive MobileBrick est en zip64 : `unzip` (MSYS) n'extrait rien sans signaler d'erreur. Utiliser `C:\Windows\System32\tar.exe -xf`.

Les archives d'origine sont conservées dans `datasets/_downloads/`. Elles peuvent être supprimées pour libérer environ 20 Go.

---

## A8 — Matrice des plateformes

| Composant | Windows 10/11 + NVIDIA | Ubuntu 22.04 / 24.04 + NVIDIA | macOS Apple Silicon |
|---|---|---|---|
| torch | 2.7 cu128 | 2.7 cu128 | 2.7 (MPS) |
| Pilote | ≥ R570 | ≥ R570 (paquet système) | — |
| VGGT | bf16 | bf16 | fp16 (à valider), float64 → processeur |
| SAM 2.1 | ✔ (extension CUDA ignorée) | ✔ | ✔ (MPS) |
| BiRefNet | ✔ | ✔ | ✔ |
| pycolmap (ajustement) | wheel (à vérifier) | wheel | wheel |
| Reconstruction dense | COLMAP CUDA (binaire officiel) | COLMAP CUDA (à compiler) | **OpenMVS** (CPU) |
| Open3D | ✔ | ✔ | ✔ |
| Hunyuan3D-2mv | fp16, FlashVDM | fp16, FlashVDM | MPS, lent (option) |
| CAD-Recode (phase 2) | sdpa | sdpa | MPS |
| CadQuery / OCP (phase 2) | wheel | wheel | wheel arm64 (à vérifier) |
| Paquet | NSIS (electron-builder) | AppImage | `.dmg` non signé |
| Intégration continue | GitHub Actions (CPU) | GitHub Actions (CPU) | GitHub Actions macOS arm64 (CPU) |
| Tests GPU | Manuels : 3060 Laptop et 4060 | Manuels (si machine dispo) | Manuels (si Mac dispo) |

---

## A9 — Premier lancement (installeur léger)

```mermaid
sequenceDiagram
    autonumber
    participant U as Utilisateur
    participant E as Electron (principal)
    participant N as Réseau (HF / miroirs)
    participant FS as Disque local

    U->>E: lance Snap2CAD (1re fois)
    E->>E: détecte OS, GPU, pilote, VRAM → profil full / degraded
    alt pilote NVIDIA < R570 ou GPU non supporté
        E-->>U: message bloquant + lien vers le pilote
    end
    E->>FS: python-build-standalone 3.11 (embarqué dans l'installeur)
    E->>N: uv sync (torch cu128 ou MPS, dépendances verrouillées)
    N-->>FS: environnement Python
    E->>N: GET manifest.json des poids
    loop pour chaque groupe (Référence, Complétion, Éditer)
        E->>N: GET avec en-tête Range (reprise)
        N-->>FS: morceaux
        E->>FS: vérification SHA-256
    end
    E->>FS: conversion VGGT fp32 → bf16 (une fois)
    E->>E: lance le worker (127.0.0.1, port aléatoire, jeton)
    E-->>U: prêt — fonctionnement 100 % hors ligne ensuite
```
