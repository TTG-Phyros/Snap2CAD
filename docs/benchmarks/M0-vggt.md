# M0 — VGGT : VRAM et temps selon le nombre d'images

Mesures brutes : [`vggt.json`](vggt.json), produites par `tools/bench_vggt.py`.
Conditions : VGGT-1B, agrégateur en bf16 (poids convertis), têtes en fp32 (elles tournent hors autocast en amont), torch 2.7.1+cu128, Windows 11. Temps = une passe avant, sans chargement ni prétraitement.

## Règle de budget (révisée sur la RTX 4060)

**Sous Windows, le pic alloué ne suffit pas.** Avec le pilote WDDM, `cudaMalloc` n'échoue pas quand la VRAM est pleine : il déborde en mémoire partagée. L'allocateur de torch ne libère son cache qu'après un échec d'allocation, il ne le rend donc jamais. La mémoire **réservée** dépasse alors la VRAM physique alors que l'alloué paraît tenir. Rien ne plante, mais tout ralentit ×2 à ×3, avec des temps erratiques.

Mesures sur la 4060, DTU scan1, modèle complet, sans plafond :

| N | alloué | réservé | temps sans plafond | temps avec plafond |
|---:|---:|---:|---:|---:|
| 8 | 5 164 Mo | 8 034 Mo | 2,4 s | 1,9 s |
| 24 | 5 796 Mo | 9 078 Mo | 21 à 30 s | 7,4 s |
| 32 | 6 109 Mo | 8 860 Mo | 27 à 57 s | 11,2 s |

`max_split_size_mb` et `garbage_collection_threshold` aggravent la situation, et `expandable_segments` n'existe pas sous Windows.

**Parade (`device.cap_cuda_memory`, appelé par `device.detect`) :** chaque processus plafonne sa VRAM à *VRAM libre au lancement − 256 Mo* (`torch.cuda.set_per_process_memory_fraction`). Le contexte CUDA et ce qu'occupe déjà le bureau sont ainsi décomptés. L'allocateur vide son cache au lieu de déborder, et un vrai dépassement devient une `OutOfMemoryError` explicite. Le banc applique ce plafond : un N « tient » s'il passe sans OOM. Il mesure aussi le même modèle que l'étape `poses_init` (sans tête de points ni de suivi).

## RTX 4060 8 Go (profil `full`)

Écran branché sur la carte : environ 1,1 Go déjà pris au lancement (bureau et contexte CUDA), soit un plafond de **6 851 Mo** (bureau léger, ~400 Mo dans `nvidia-smi`).

| N images | kitchen 518×350 : temps | alloué | réservé | DTU scan1 518×392 : temps | alloué | réservé |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0,8 s | 3 623 Mo | 5 020 Mo | 0,9 s | 3 730 Mo | 5 346 Mo |
| 8 | 1,7 s | 4 544 Mo | 6 508 Mo | 1,8 s | 4 771 Mo | 6 146 Mo |
| 12 | 2,8 s | 4 672 Mo | 6 732 Mo | 2,8 s | 4 909 Mo | 6 402 Mo |
| 16 | 3,7 s | 4 804 Mo | 6 284 Mo | 4,4 s | 5 058 Mo | 6 618 Mo |
| 20 | 5,1 s | 4 931 Mo | 6 494 Mo | 5,5 s | 5 199 Mo | 6 408 Mo |
| 24 | — | — | — | 7,6 s | 5 353 Mo | 6 670 Mo |
| 25 | 6,9 s | 5 097 Mo | 6 802 Mo | — | — | — |
| 28 | — | — | — | 9,3 s | 5 491 Mo | 6 476 Mo |
| 32 | — | — | — | 10,7 s | 5 641 Mo | 6 590 Mo |
| 36 | — | — | — | 12,8 s | 5 781 Mo | 6 420 Mo |
| 40 | — | — | — | OOM | | |

Le plafond suit l'occupation du bureau, qui varie de 400 à 850 Mo selon les applications ouvertes. Avec un plafond réduit de 400 Mo (6 451 Mo), N = 28 passe (9 s, 6 256 Mo réservés) et N = 32 tombe en OOM.

**N₈ = 28** pour des photos 4:3. C'est la valeur qui tient même avec un bureau chargé ; on atteint 36 avec un bureau léger.

Observations :
- Avec le plafond, le temps croît presque linéairement : ~0,35 s par image au-delà de 8. Le « ~N² » vu sur la 3060 venait surtout du débordement en mémoire partagée.
- L'OOM arrive avant que l'alloué n'atteigne le plafond (N = 40 : ~5,9 Go alloués pour 6,85 Go de plafond). La fragmentation de l'allocateur coûte 0,6 à 1 Go.

## RTX 3060 Laptop 6 Go (profil `degraded`) — à remesurer

Mesures faites avant la règle du plafond : modèle complet (avec tête de points), budget théorique 5 632 Mo comparé au seul pic alloué. Le réservé n'a pas été relevé : les temps comprennent probablement du débordement en mémoire partagée.

| N images | kitchen 518×350 : temps | alloué | DTU scan1 518×392 : temps | alloué |
|---:|---:|---:|---:|---:|
| 4 | 1,2 s | 4 010 Mo | 1,4 s | 4 117 Mo |
| 8 | 4,3 s | 4 936 Mo | 5,5 s | 5 164 Mo |
| 12 | 7,0 s | 5 070 Mo | 9,8 s | 5 309 Mo |
| 16 | 10,0 s | 5 219 Mo | 13,1 s | 5 477 Mo |
| 20 | 14,7 s | 5 351 Mo | 21,2 s | 5 622 Mo |
| 24 | — | — | 30,2 s | 5 796 Mo ✗ |
| 25 | 26,4 s | 5 535 Mo | — | — |

N₆ = 20 retenu à l'époque. **Avec le plafond, il faut le remesurer** : la marge de fragmentation (0,6 à 1 Go) peut provoquer un OOM sous 20 images. Commande à lancer sur le portable :

```bash
uv run --project worker python tools/bench_vggt.py dtu_scan1 "datasets/dtu/SampleSet/MVS Data/Rectified/scan1/rect_*_3_r5000.png" 4 8 12 16 20 24
```

Observations communes :
- ~3,2 Go de poids en VRAM (agrégateur bf16 1,8 Go + têtes fp32 1,4 Go ; un peu moins sans la tête de points). Le saut entre 4 et 8 images vient de la tête DPT, qui traite les vues par paquets de 8.
- Au-delà de 8 images, ~35 Mo alloués par image.
- Pas de FlashAttention dans torch pour Windows : SDPA utilise le noyau « memory-efficient ». Linux devrait être plus rapide.
- Pistes si N ne suffit pas : passer les têtes en bf16 (~0,7 Go gagnés), à valider en précision. Sinon traitement par lots + ajustement global (plan R5).

## Précision bf16 vs fp32 (4 images kitchen)

Écart moyen entre notre chargement bf16 et la recette amont (poids fp32 + autocast bf16) : poses 1,6·10⁻⁴ (encodage ~0,4), profondeur 0,3 % relatif. Négligeable pour des poses initiales, affinées ensuite par l'ajustement de faisceaux.
