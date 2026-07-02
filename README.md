# emotion_recognition_project



# Classification d'Émotions FER2013 - ResNet18 Baseline

## Description

Ce projet vise à entraîner une architecture **ResNet18** sur le dataset **FER2013** (`msambare/fer2013`) pour la reconnaissance d'expressions faciales.

## Pipeline Technique

1. **Prétraitement :** Redimensionnement, normalisation et Data Augmentation (RandomHorizontalFlip, RandomRotation).
2. **Gestion du Déséquilibre :** Calcul des poids de classes pour équilibrer la `CrossEntropyLoss`.
3. **Modélisation :** Utilisation de `torchvision.models.resnet18` (transfer learning ou entraînement scratch).
4. **Interprétabilité :** Utilisation de **Grad-CAM** pour visualiser les zones d'intérêt du modèle lors de la prédiction.

## Métriques

* [ ] **Matrice de Confusion :** Analyse des erreurs systématiques.
* [ ] **Rapport de Classification :** Précision, Recall, F1-Score par émotion.
* [ ] **Analyse Critique :** Identification automatique des classes les plus difficiles.

## Tester le modèle en local

Le modèle entraîné (`resnet18_fer2013.pth.zip`) peut être testé localement via `main.py`.

### Installation

```bash
pip install -r requirements.txt
```

### Configuration (optionnel)

Le chemin du modèle et quelques paramètres se règlent via variables d'environnement (voir `.env.example`) :

```bash
cp .env.example .env
```

| Variable       | Défaut                        | Description                          |
|----------------|--------------------------------|---------------------------------------|
| `MODEL_PATH`   | `resnet18_fer2013.pth.zip`     | Chemin vers le checkpoint             |
| `DEVICE`       | *(auto)*                       | Forcer `cpu` ou `cuda`                |
| `CAMERA_INDEX` | `0`                             | Index de la webcam                    |
| `IMG_SIZE`     | `224`                           | Taille de redimensionnement des images |

### Lancement

```bash
# Une seule image
python main.py --image chemin/vers/photo.jpg

# Toutes les images d'un dossier
python main.py --dir chemin/vers/dossier

# Évaluation complète sur un dataset structuré par classe (comme FER2013)
python main.py --test-dir ./Datasets/test

# Démo temps réel via la webcam
python main.py --webcam
```

