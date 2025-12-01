# Projet : Diagnostic de Maladie (Classification)

## 📋 Contexte

Projet de 4ème année, option IA & Data Science. Ce projet vise à développer, entraîner et évaluer un modèle de classification pour prédire la présence ou l'absence d'une maladie à partir de données médicales.

**Jeu de données utilisé :** Pima Indian Diabetes Dataset
- Variables : Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Variable cible : Outcome (0 = absence de diabète, 1 = présence de diabète)

## 🎯 Objectifs

- Développer et évaluer un modèle de classification performant
- Réaliser une analyse complète des données (EDA univariée, bivariée, multivariée)
- Effectuer un pré-traitement rigoureux des données
- Comparer différents modèles de machine learning
- Fournir une interprétation clinique des résultats
- Produire un livrable reproductible et documenté

## 📁 Structure du Projet

```
project_diagnostic_maladie/
├── data/
│   ├── raw/                    # Jeux de données originaux
│   ├── interim/                # Versions nettoyées partielles
│   └── processed/              # Données prêtes pour modélisation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── src/
│   ├── data/                   # Scripts de chargement et nettoyage
│   ├── features/               # Ingénierie des features
│   ├── models/                 # Entraînement et sauvegarde des modèles
│   └── evaluation/             # Métriques et visualisation
├── reports/
│   ├── figures/                # Graphiques et visualisations
│   ├── literature_review.md    # Synthèse bibliographique
│   ├── pca_analysis.md         # Analyse en composantes principales
│   └── final_report.md         # Rapport final du projet
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🛠️ Installation et Configuration

### Prérequis

- Python 3.11
- Visual Studio Code
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Configuration VS Code (Recommandée)

**Extensions recommandées :**
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)
- autoDocstring (pour documenter le code)

**Créer un environnement virtuel :**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Puis installer les dépendances :
```bash
pip install -r requirements.txt
```

### Dépendances principales

- `pandas` - Manipulation de données
- `numpy` - Calculs numériques
- `scikit-learn` - Machine learning
- `matplotlib` & `seaborn` - Visualisation
- `scipy` - Statistiques
- `xgboost` - Modèles d'ensemble avancés
- `shap` - Interprétabilité des modèles
- `jupyterlab` - Environnement notebooks

## 🚀 Démarrage Rapide

1. **Cloner le projet**
   ```bash
   git clone <url-du-depot>
   cd project_diagnostic_maladie
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Télécharger les données**
   - Placer le fichier CSV dans `data/raw/`
   - Source : [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

4. **Lancer Jupyter Lab**
   ```bash
   jupyter lab
   ```

5. **Suivre les notebooks dans l'ordre**
   - `01_data_exploration.ipynb` → Exploration des données
   - `02_preprocessing.ipynb` → Nettoyage et préparation
   - `03_feature_engineering.ipynb` → Création de features
   - `04_modeling_and_evaluation.ipynb` → Modélisation et évaluation

## 📊 Méthodologie

### Phase 1 : Recherche Bibliographique (2-3 jours)
- Collecte de 5+ articles scientifiques pertinents
- Extraction des méthodologies et métriques utilisées
- Synthèse comparative dans `reports/literature_review.md`

### Phase 2 : Analyse Exploratoire des Données (6-10 jours)

**Analyse Univariée**
- Statistiques descriptives (mean, median, std, skewness, kurtosis)
- Gestion des valeurs manquantes
- Détection et traitement des valeurs aberrantes
- Visualisations (histogrammes, boxplots, QQ-plots)

**Analyse Bivariée**
- Corrélations entre variables (X vs X)
- Relations avec la variable cible (X vs Y)
- Tests statistiques (ANOVA, t-test, chi2)
- Visualisations (heatmaps, scatter matrices, boxplots groupés)

**Analyse Multivariée**
- Analyse en Composantes Principales (PCA)
- Réduction de dimensionnalité
- Interprétation des composantes principales

### Phase 3 : Modélisation (4-6 jours)

**Modèles testés**
- Baseline : Logistic Regression
- Ensemble : Random Forest, XGBoost
- Optionnels : SVM, KNN, Neural Networks

**Processus**
- Split train/validation/test (60/20/20)
- Pipelines sklearn pour reproductibilité
- Grid Search / Random Search pour hyperparamètres
- Validation croisée stratifiée

**Métriques d'évaluation**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, courbe Precision-Recall
- Matrice de confusion
- Analyse des faux positifs/négatifs

### Phase 4 : Interprétation Clinique
- Analyse SHAP pour l'interprétabilité
- Importance des features
- Implications médicales des résultats
- Recommandations de seuils opératoires

## 📈 Résultats Attendus

Les résultats complets seront documentés dans `reports/final_report.md` et incluront :

- Synthèse de l'analyse exploratoire
- Comparaison des modèles testés
- Métriques de performance sur le jeu de test
- Interprétation clinique des prédictions
- Limites et perspectives d'amélioration

## 🔬 Reproductibilité

- Tous les seeds aléatoires sont fixés pour garantir la reproductibilité
- Les pipelines sklearn permettent de rejouer l'ensemble du workflow
- Le modèle final est sauvegardé dans `models/final_model.joblib`

## ✅ Checklist de Progression

- [ ] Arborescence créée et dépendances installées
- [ ] 5 articles collectés et synthétisés
- [ ] Analyse univariée complète
- [ ] Imputation et traitement des outliers documentés
- [ ] Analyse bivariée et tests statistiques
- [ ] PCA réalisée et interprétée
- [ ] Pipelines de modélisation créés
- [ ] Modèles entraînés et comparés
- [ ] Évaluation complète (ROC, PR, confusion matrix)
- [ ] Interprétabilité (SHAP, feature importance)
- [ ] Rapport final rédigé
- [ ] Présentation préparée

## 📚 Ressources

- [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

## 👥 Auteur

Souhaib MADHOUR - 4ème année IA & Data Science

## 📅 Planning

- **Semaine 1 :** Mise en place + recherche bibliographique + EDA univariée
- **Semaine 2 :** EDA bivariée + nettoyage + feature engineering
- **Semaine 3 :** Modélisation + tuning + interprétabilité
- **Semaine 4 :** Validation + rapport final + présentation

## 📝 License

Ce projet est réalisé dans un cadre académique.

---

**Note :** Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue ou à me contacter.