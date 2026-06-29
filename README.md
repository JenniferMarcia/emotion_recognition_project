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

