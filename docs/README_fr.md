# Pipeline de Restauration d'Images

Pipeline reproductible pour l'entraînement et l'évaluation de modèles de restauration d'images sur données synthétiques (tampons, flou, bruit, occultations).

---

## Table des matières

1. [Installation](#installation)
2. [Structure du projet](#structure-du-projet)
3. [Génération des données](#1-génération-des-données-synthétiques)
4. [Entraînement](#2-entraînement-dun-modèle)
5. [Évaluation](#3-évaluation-dune-expérience)
6. [Fine-tuning](#4-fine-tuning--recherche-dhyperparamètres)
7. [Visualisation des résultats](#5-visualisation-des-résultats-de-tuning)
8. [Modèles disponibles](#modèles-disponibles)
9. [Métriques](#métriques)
10. [Configurer un espace de recherche](#configurer-un-espace-de-recherche-personnalisé)

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note GPU :** TensorFlow utilisera automatiquement le GPU si les drivers CUDA sont installés.
> Les messages CUDA sur CPU sont normaux et n'affectent pas l'exécution.

---

## Structure du projet

```
image_restoration/
│
├── run.py                        ← point d'entrée unique (toutes les commandes)
├── show_results.py               ← affiche un tableau de résultats de tuning
├── requirements.txt
├── README.md
│
├── configs/                      ← espaces de recherche pour le tuning (YAML)
│   ├── tune_unet.yaml
│   ├── tune_unet_grid.yaml
│   └── tune_vit.yaml
│
├── data/
│   └── synthetic/                ← généré par `python run.py generate`
│       ├── X_noisy.npy           ← images corrompues  (N, 128, 128, 1)
│       └── X_clean.npy           ← images propres     (N, 128, 128, 1)
│
├── experiments/                  ← créé automatiquement à l'entraînement
│   ├── <exp_name>/
│   │   ├── config.json
│   │   ├── train.log
│   │   ├── metrics.json
│   │   ├── weights/
│   │   │   ├── best_weights.weights.h5
│   │   │   └── final_weights.weights.h5
│   │   └── outputs/comparisons/
│   │       ├── comparison.png
│   │       └── loss_curve.png
│   │
│   └── tuning/
│       └── <tuning_name>/
│           ├── trial_000/
│           ├── trial_001/ …
│           ├── results.json
│           ├── best_command.sh
│           └── tuning.log
│
├── src/
│   ├── data_generator.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── models/
│       ├── cnn.py
│       ├── unet.py
│       ├── unet_gan.py
│       ├── mwcnn.py
│       └── vit.py
│
└── tuning/
    ├── search_strategies.py      ← RandomSearch et GridSearch
    └── tuner.py                  ← orchestrateur (appelle trainer pour chaque trial)
```

---

## 1. Génération des données synthétiques

Le pipeline génère des paires d'images **(bruitée → propre)** à partir de zéro.
Chaque image contient des chiffres sur fond papier, avec différents types de corruption.

```bash
python run.py generate \
  --num_samples 1000 \
  --img_size    128  \
  --noise_level 0.06 \
  --corruption  stamp \
  --output      data/synthetic/
```

### Types de corruption

| Option      | Effet                                                         |
|-------------|---------------------------------------------------------------|
| `stamp`     | Tampon elliptique semi-transparent (défaut)                   |
| `blur`      | Flou gaussien de noyau aléatoire (3×3, 5×5 ou 7×7)           |
| `occlusion` | Rectangle de couleur papier couvrant ~20 % de l'image         |
| `all`       | Les trois corruptions appliquées simultanément                |

---

## 2. Entraînement d'un modèle

```bash
python run.py train \
  --model        unet \
  --dataset      data/synthetic \
  --exp_name     exp_unet_01 \
  --epochs       10 \
  --batch_size   16 \
  --lr           0.001 \
  --optimizer    adam \
  --loss         auto \
  --patience     8 \
  --dropout      0.3 \
  --base_filters 32
```

### Paramètres spécifiques au ViT

```bash
python run.py train --model vit \
  --patch_size 8  \
  --embed_dim  256 \
  --num_heads  8   \
  --mlp_dim    512 \
  --num_layers 6
```

### Exemples par modèle

```bash
# CNN (le plus rapide)
python run.py train --model cnn --epochs 5 --batch_size 16 --lr 0.001 \
  --dataset data/synthetic --exp_name exp_cnn

# U-Net (recommandé en premier)
python run.py train --model unet --epochs 10 --batch_size 16 --lr 0.001 \
  --dropout 0.2 --base_filters 32 --dataset data/synthetic --exp_name exp_unet

# MWCNN (wavelet loss auto)
python run.py train --model mwcnn --epochs 10 --batch_size 16 --lr 0.001 \
  --dataset data/synthetic --exp_name exp_mwcnn

# ViT (config légère)
python run.py train --model vit --epochs 5 --batch_size 8 --lr 0.0005 \
  --embed_dim 128 --num_heads 4 --num_layers 2 --mlp_dim 256 \
  --dataset data/synthetic --exp_name exp_vit

# GAN adversarial
python run.py train --model unet_gan --epochs 10 --batch_size 8 --lr 0.0002 \
  --dataset data/synthetic --exp_name exp_gan
```

---

## 3. Évaluation d'une expérience

```bash
python run.py evaluate \
  --exp     experiments/exp_unet_01/ \
  --dataset data/synthetic \
  --n_eval  200
```

Le script relit `config.json`, recharge les poids, calcule PSNR/SSIM/MAE
et sauvegarde `eval_metrics.json` et `eval_comparison.png`.

---

## 4. Fine-tuning / Recherche d'hyperparamètres

Le tuner lance plusieurs expériences avec des configurations différentes,
compare les résultats via SSIM, et identifie la meilleure combinaison.

### Commandes

```bash
# Recherche aléatoire avec espace par défaut intégré
python run.py tune \
  --model       unet \
  --trials      10 \
  --strategy    random \
  --tune_epochs 3 \
  --dataset     data/synthetic \
  --tuning_name tune_unet_quick

# Recherche aléatoire avec config YAML
python run.py tune \
  --config      configs/tune_unet.yaml \
  --trials      10 \
  --tuning_name tune_unet_yaml

# Recherche par grille (grid search)
python run.py tune \
  --config      configs/tune_unet_grid.yaml \
  --strategy    grid \
  --trials      20 \
  --tune_epochs 3 \
  --tuning_name tune_unet_grid

# Tuning du ViT
python run.py tune \
  --config      configs/tune_vit.yaml \
  --trials      8 \
  --tune_epochs 2 \
  --tuning_name tune_vit
```

### Paramètres disponibles pour `tune`

| Paramètre       | Défaut                    | Description                                            |
|-----------------|---------------------------|--------------------------------------------------------|
| `--model`       | `unet`                    | Architecture à optimiser                               |
| `--trials`      | `10`                      | Nombre de configurations à tester                      |
| `--strategy`    | `random`                  | `random` (aléatoire) ou `grid` (exhaustif)             |
| `--config`      | espace par défaut intégré | Fichier YAML définissant l'espace de recherche         |
| `--tuning_name` | `tune_<model>_<strategy>` | Nom → `experiments/tuning/<name>/`                     |
| `--tune_epochs` | `3`                       | Époques par trial (garder ≤ 5 pour la rapidité)        |
| `--dataset`     | `data/synthetic`          | Chemin vers les données                                |
| `--seed`        | `42`                      | Graine aléatoire (reproductibilité)                    |

### Stratégies

**`random` (recommandée)** — Échantillonne des configurations au hasard.
Très efficace sur des espaces larges ou continus (lr log-uniforme, dropout continu…).

**`grid`** — Énumère toutes les combinaisons (produit cartésien).
À utiliser avec peu de paramètres et des valeurs discrètes uniquement (`type: choice`).
Si le nombre de combinaisons dépasse `--trials`, un sous-échantillon est tiré.

---

## 5. Visualisation des résultats de tuning

```bash
# Tous les trials triés par SSIM
python show_results.py experiments/tuning/tune_unet_quick/results.json

# Top 5 triés par PSNR
python show_results.py experiments/tuning/tune_unet_quick/results.json \
  --sort psnr --top 5
```

### Entraîner la meilleure config trouvée

```bash
# Option A — exécuter le script généré automatiquement
bash experiments/tuning/tune_unet_quick/best_command.sh

# Option B — augmenter les époques manuellement avec les params trouvés
python run.py train \
  --model unet \
  --epochs 30 \
  --lr 0.0008 \
  --batch_size 16 \
  --dropout 0.15 \
  --base_filters 32 \
  --dataset data/synthetic \
  --exp_name exp_unet_best_final
```

---

## Modèles disponibles

### 1. `cnn` — CNN résiduel (Baseline)

**Catégorie :** Réseau convolutif pur, apprentissage local

**Architecture :** Extraction de features → N blocs résiduels (Conv + BN + ReLU + skip connection) → projection vers l'image de sortie.

**Points forts :**
- Le plus rapide à entraîner
- Sert de référence pour évaluer les autres modèles
- Peu de paramètres, idéal pour des tests rapides

**Paramètres clés :** `--base_filters` (défaut 64), profondeur fixée à 6 blocs

---

### 2. `unet` — U-Net (Encodeur-Décodeur)

**Catégorie :** Architecture encodeur-décodeur avec skip connections

**Architecture :** 4 niveaux d'encodage (Conv + MaxPooling) → goulot d'étranglement → 4 niveaux de décodage (ConvTranspose + Concatenate avec l'encodeur correspondant).

**Points forts :**
- Architecture de référence en restauration d'images
- Les skip connections préservent les détails fins (contours, textures)
- Bon équilibre qualité / vitesse
- **Recommandé comme premier modèle à tester**

**Paramètres clés :** `--base_filters` (32 → 64 → 128 → 256 → 512), `--dropout`

---

### 3. `mwcnn` — Multi-level Wavelet CNN

**Catégorie :** CNN dans le domaine des ondelettes, multi-résolution

**Architecture :** Décomposition par transformée de Haar (DWT) sur 3 niveaux en sous-bandes fréquentielles (LL/LH/HL/HH) → blocs CNN résiduels à chaque résolution → reconstruction par IWT inverse.

**Points forts :**
- Capture explicitement les composantes basse et haute fréquence
- Adapté aux images avec bruit structuré et textures fines
- Loss multi-échelle dédiée (`wavelet_loss`) sélectionnée automatiquement

**Contrainte :** `--img_size` doit être un multiple de 8

---

### 4. `unet_gan` — GAN adversarial (UDBNet / Pix2Pix)

**Catégorie :** Réseau génératif adversarial (GAN)

**Architecture :**
- **Générateur :** U-Net 7 niveaux (stride-2), sortie tanh en `[-1, 1]`
- **Discriminateur :** PatchGAN — évalue la réalité de patches locaux
- **Loss :** BCE adversariale + λ×L1 (λ=100) pour la fidélité globale

**Points forts :**
- Produit des images visuellement réalistes
- Capable d'halluciner des détails plausibles
- Nécessite plus d'époques pour converger (~20–50)

**Particularité :** Deux optimiseurs (générateur + discriminateur), boucle d'entraînement personnalisée.

---

### 5. `vit` — Vision Transformer

**Catégorie :** Architecture basée sur l'attention (Transformer)

**Architecture :**
- **PatchEmbed :** découpe l'image en patches → tokens de dimension `embed_dim`
- **Positional Embedding :** appris pour chaque position
- **N blocs Transformer :** LayerNorm → MultiHeadAttention → résidu → MLP (GELU) → résidu
- **Tête CNN de décodage :** 3× ConvTranspose2D pour remonter à la résolution originale

**Points forts :**
- Capture les dépendances globales dans l'image (attention sur tous les patches)
- Très flexible architecturalement
- Pertinent pour étudier les représentations longue portée

**Paramètres clés :** `--patch_size`, `--embed_dim`, `--num_heads`, `--num_layers`, `--mlp_dim`

---

### Tableau récapitulatif

| Modèle      | Famille           | Loss par défaut       | Vitesse | Recommandé pour                       |
|-------------|-------------------|-----------------------|---------|---------------------------------------|
| `cnn`       | CNN pur           | weighted MSE + SSIM   | ⚡⚡⚡   | Baseline rapide, premiers tests        |
| `unet`      | Encodeur-décodeur | weighted MSE + SSIM   | ⚡⚡     | Usage général, bon point de départ     |
| `mwcnn`     | Ondelettes        | Wavelet multi-échelle | ⚡⚡     | Bruit structuré, haute fréquence       |
| `unet_gan`  | GAN adversarial   | BCE + λ·L1            | ⚡       | Rendu réaliste, images complexes       |
| `vit`       | Transformer       | weighted MSE + SSIM   | ⚡       | Dépendances globales, expérimentation  |

---

## Métriques

| Métrique | Description                         | Interprétation   | Plage typique |
|----------|-------------------------------------|------------------|---------------|
| **PSNR** | Peak Signal-to-Noise Ratio (en dB)  | Plus élevé = mieux | 25 – 40 dB  |
| **SSIM** | Similarité structurelle             | Plus proche de 1 | 0.7 – 0.99    |
| **MAE**  | Erreur absolue moyenne par pixel    | Plus bas = mieux | 0.01 – 0.15   |

La métrique primaire pour comparer les trials de tuning est le **SSIM**.

---

## Configurer un espace de recherche personnalisé

Les fichiers YAML dans `configs/` définissent quels hyperparamètres explorer.

### Structure d'un fichier YAML

```yaml
# Paramètres fixes pour tous les trials
base:
  dataset:    data/synthetic
  epochs:     3
  img_size:   128
  patience:   5

# Espace de recherche
space:
  lr:
    type: log_uniform     # échantillonnage log-uniforme
    low:  0.0001
    high: 0.01

  batch_size:
    type: choice          # valeurs discrètes
    values: [8, 16, 32]

  dropout:
    type: uniform         # continu linéaire
    low:  0.1
    high: 0.4

  optimizer:
    type: choice
    values: [adam, adamw]
```

### Types de paramètres

| Type          | Usage                              | Compatible grid ? |
|---------------|------------------------------------|-------------------|
| `choice`      | Valeurs discrètes explicites       | ✅ Oui             |
| `log_uniform` | Plage continue échelle log (lr…)   | ❌ Random only     |
| `uniform`     | Plage continue échelle linéaire    | ❌ Random only     |
| `int`         | Entier dans un intervalle          | ❌ Random only     |

> Pour la **grid search**, tous les paramètres doivent être de type `choice`.

### Élargir l'espace de recherche — exemples

```yaml
# Plus de valeurs de learning rate
lr:
  type: choice
  values: [0.0001, 0.0003, 0.001, 0.003, 0.01]

# Tester plus d'architectures U-Net
base_filters:
  type: choice
  values: [8, 16, 32, 64]

dropout:
  type: choice
  values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Comparer les fonctions de loss
loss:
  type: choice
  values: [weighted_mse_ssim, wavelet, mse, mae]

# Paramètres ViT
embed_dim:
  type: choice
  values: [64, 128, 256, 512]

num_layers:
  type: choice
  values: [2, 4, 6, 8]

num_heads:
  type: choice
  values: [4, 8]           # embed_dim doit être divisible par num_heads

mlp_dim:
  type: choice
  values: [128, 256, 512, 1024]
```

---

### Tuner les coefficients de la fonction de loss

La fonction de loss principale (`weighted_mse_ssim`) contient **3 coefficients internes** qui ont un impact direct sur ce que le modèle optimise. Ils sont exposés comme hyperparamètres à part entière dans le système de tuning.

#### Comprendre les 3 coefficients

```python
def make_weighted_mse_ssim(ink_weight, mse_alpha, ssim_alpha):
    def loss(y_true, y_pred):
        weight    = 1.0 + ink_weight * (1.0 - y_true)   # ← coefficient 1
        mse_loss  = tf.reduce_mean(weight * tf.square(y_true - y_pred))
        ssim_loss = 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0)
        return mse_alpha * mse_loss + ssim_alpha * ssim_loss  # ← coefficients 2 et 3
```

| Coefficient  | Défaut | Rôle                                                                 |
|--------------|--------|----------------------------------------------------------------------|
| `ink_weight` | `4.0`  | Poids supplémentaire accordé aux **pixels sombres (encre)**. Plus il est élevé, plus le modèle pénalise les erreurs sur les chiffres plutôt que sur le fond papier. |
| `mse_alpha`  | `0.5`  | Poids de la composante **MSE** dans la loss finale.                  |
| `ssim_alpha` | `0.5`  | Poids de la composante **SSIM** dans la loss finale. En pratique, `mse_alpha + ssim_alpha = 1` est la convention naturelle mais ce n'est pas imposé. |

#### Effets pratiques à connaître

- **`ink_weight` élevé (ex: 8.0)** → le modèle se concentre davantage sur la qualité de reconstruction des chiffres, au détriment du fond. Utile quand les chiffres sont fins ou peu contrastés.
- **`ink_weight` faible (ex: 1.0)** → traitement uniforme de tous les pixels. Plus proche d'un MSE standard.
- **`mse_alpha` élevé (ex: 0.8)** → favorise la fidélité pixel-à-pixel, résultats plus nets mais parfois moins naturels.
- **`mse_alpha` faible (ex: 0.2)** → favorise la similarité structurelle globale (SSIM), résultats perceptuellement plus agréables mais parfois légèrement flous.

#### Ajouter ces coefficients dans un fichier YAML

```yaml
base:
  dataset:    data/synthetic
  epochs:     3
  loss:       weighted_mse_ssim   # obligatoire pour activer la loss paramétrique

space:
  lr:
    type: log_uniform
    low:  0.0001
    high: 0.01

  # ── Coefficients de la loss ──────────────────────────────────────────
  loss:
    type: choice
    values: [weighted_mse_ssim, mse]

  ink_weight:
    type: choice
    values: [1.0, 2.0, 4.0, 6.0, 8.0]   # tester un spectre large en premier

  mse_alpha:
    type: choice
    values: [0.3, 0.5, 0.7, 0.9]         # ssim_alpha = 1 - mse_alpha implicitement

  # ssim_alpha peut aussi être exploré indépendamment si besoin :
  # ssim_alpha:
  #   type: choice
  #   values: [0.1, 0.3, 0.5]
```

> **Important :** `loss: weighted_mse_ssim` doit être fixé dans `base:` pour que le trainer
> active la version paramétrique. Si `loss` est dans `space:` avec d'autres valeurs comme `mse`,
> les coefficients `ink_weight` / `mse_alpha` seront ignorés pour ces trials (comportement normal).

#### Exemple de config complète orientée loss

```yaml
# configs/tune_loss_coefficients.yaml
# Recherche focalisée sur l'impact des coefficients de loss — U-Net architecture fixe

base:
  dataset:      data/synthetic
  epochs:       5
  model:        unet
  loss:         weighted_mse_ssim
  lr:           0.001
  batch_size:   16
  base_filters: 32
  dropout:      0.2

space:
  loss:
      type: choice
      values: [weighted_mse_ssim, mse]


  ink_weight:
    type: uniform
    low:  1.0
    high: 10.0

  mse_alpha:
    type: uniform
    low:  0.2
    high: 0.9
```

```bash
# Lancer la recherche
python run.py tune \
  --config      configs/tune_loss_coefficients.yaml \
  --trials      12 \
  --strategy    random \
  --tune_epochs 5 \
  --tuning_name tune_loss_coeffs

# Voir quels coefficients donnent le meilleur SSIM
python show_results.py experiments/tuning/tune_loss_coeffs/results.json --sort ssim
```

---

## Workflow complet recommandé

```bash
# 1. Générer les données
python run.py generate --num_samples 1000 --corruption all --output data/synthetic/

# 2. Tester tous les modèles
for model in cnn unet mwcnn vit; do
  python run.py train --model $model --epochs 10 \
    --dataset data/synthetic --exp_name exp_$model
done

# 3. Évaluer tous les modèles
for model in cnn unet mwcnn vit; do
  python run.py evaluate --exp experiments/exp_$model/ --n_eval 200
done

# 4. Optimiser le meilleur modèle
python run.py tune --config configs/tune_unet.yaml \
  --trials 15 --tune_epochs 5 --tuning_name tune_unet_final

# 5. Voir les résultats du tuning
python show_results.py experiments/tuning/tune_unet_final/results.json --top 5

# 6. Entraîner la meilleure config avec plus d'époques
bash experiments/tuning/tune_unet_final/best_command.sh
```