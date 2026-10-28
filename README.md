# 🏃 IA Sport

**IA Sport** est une preuve de concept (POC) d'AutoML guidé par un LLM, appliquée aux données sportives.
> 📄 Ce projet a fait l'objet d'une publication à **ICAART 2026** : *Context-Aware AI-Based Framework for Sports Analytics* ([DOI : 10.5220/0014469700004052](https://doi.org/10.5220/0014469700004052)). Voir la section [Publication](#-publication).

L'utilisateur fournit un dataset et décrit son objectif en langage naturel (par exemple *« Prédire la probabilité de blessure des joueurs pendant l'entraînement »*) ; le système détecte automatiquement le type de tâche, la variable cible et les variables pertinentes, puis entraîne et compare plusieurs pipelines de prétraitement et modèles de Machine Learning.

---

## ✨ Fonctionnalités

- **Contextualisation par LLM** : un agent RAG (OpenAI + recherche web Tavily) analyse le prompt et les colonnes du dataset pour identifier :
  - le type de tâche (classification, régression, séries temporelles…),
  - le cas d'usage sportif,
  - la colonne cible et les features pertinentes.
- **Pipelines de prétraitement** : plusieurs pipelines prédéfinis (imputation, normalisation, encodage, PCA/LDA, rééchantillonnage…) et génération de pipelines aléatoires.
- **Comparaison de modèles** :
  - *Classification* : XGBoost, LightGBM, CatBoost, Random Forest, MLP, KNN, SGD, Decision Tree, Balanced Bagging.
  - *Régression* : XGBoost, Random Forest, MLP.
- **Base de connaissances (Weaviate)** : stocke les expériences passées (prompt, profil des données, pipeline, métriques) avec des vecteurs sémantiques pour recommander les meilleures combinaisons pipeline/modèle sur de nouveaux datasets (modes `no_kb`, `kb_low`, `kb_high`).
- **Interface web Gradio** pour charger un dataset, saisir un objectif et visualiser les résultats.
- **Déploiement Docker** (application + Weaviate) et pipeline CI/CD GitLab.

## 🎯 Cas d'usage

| ID | Cas d'usage | Cible | Type de tâche |
|----|-------------|-------|---------------|
| UC-IP | Prédiction de blessures | `injury` | Classification / séries temporelles |
| UC-RPE | Prédiction de la perception de l'effort (RPE) | `RPE` | Régression |
| UC-PRM | Prédiction des résultats de matchs | `result` | Classification |
| UC-REC | Recrutement de joueurs | `player_id` | Recommandation |

## 🏗️ Architecture

```
Dataset (CSV/Excel/JSON) + Prompt utilisateur
              │
              ▼
   DataLoader ──► Contextualization (agent RAG / LLM)
              │        → tâche, cible, features
              ▼
        ContextStack (état partagé)
              │
              ▼
   KnowledgeBase (Weaviate) ──► recommandation pipeline/modèle
              │
              ▼
   Pipelines de prétraitement ──► Entraînement & évaluation des modèles
              │
              ▼
        Résultats / métriques (CLI ou interface Gradio)
```

## 📁 Structure du projet

```
├── main.py                  # Exécution en ligne de commande (exemples blessures & RPE)
├── core/                    # Orchestration : chargement, contexte, exécution des pipelines
├── modules/                 # DataLoader, Contextualization, ContextStack, KnowledgeBase
├── models/                  # Modèles de classification et de régression
├── utils/
│   ├── pipelines/           # Pipelines de prétraitement (classification, régression, anomalies, vision)
│   ├── metrics/             # Métriques d'évaluation
│   ├── rag_feature_agent_langchain.py  # Agent LLM de détection de tâche/cible/features
│   └── Arguments.py         # Configuration (cas d'usage, hyperparamètres, modèle LLM)
├── services/                # Logs et service de base de connaissances
├── ui/                      # Application Gradio, Dockerfile, docker-compose
├── datasets/                # Jeux de données sportifs (blessures, RPE, IoT, FIFA, matchs…)
└── Exp3.ipynb               # Notebook d'expérimentations et de visualisation des résultats
```

## 🚀 Installation

### Prérequis

- Python 3.11+
- Docker (pour Weaviate)
- Une clé API OpenAI (et optionnellement une clé Tavily pour la recherche web)

### Installation locale

```bash
git clone https://github.com/<votre-utilisateur>/ia-sport.git
cd ia-sport

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Variables d'environnement

Créez un fichier `.env` à la racine (il ne doit **jamais** être commité) :

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
WEAVIATE_URL=http://localhost:8080
```

### Lancer Weaviate

```bash
bash modules/run_weaviate.sh
```

## ▶️ Utilisation

### Interface Gradio

```bash
bash run_gradio.sh        # Linux / macOS
run_gradio.bat            # Windows
```

Puis ouvrez l'URL affichée dans le terminal.

### Ligne de commande

```bash
python main.py
```

Exécute deux exemples :
- **Régression** : prédiction du RPE à partir de `datasets/iot_sports_training_dataset.csv` ;
- **Classification** : prédiction des blessures à partir de `datasets/day_approach_maskedID_timeseries.csv`.

### Docker Compose (application + Weaviate)

```bash
cd ui
docker compose up --build
```

L'application est disponible sur [http://localhost:7860](http://localhost:7860).

## 📊 Datasets

Le dossier `datasets/` contient notamment :
- données de blessures d'athlètes (approches journalière et hebdomadaire, athlètes universitaires) ;
- données d'entraînement IoT et RPE ;
- données joueurs FIFA 21/22 ;
- résultats de matchs de Premier League (saisons 2009–2019).

Ces données proviennent de sources publiques (principalement Kaggle) et sont utilisées à des fins de recherche.

## 🛠️ Technologies

Python · scikit-learn · XGBoost · LightGBM · CatBoost · imbalanced-learn · PyTorch · OpenAI · LangChain · Tavily · Weaviate · Gradio · Docker · GitLab CI

## 📚 Publication

Ce travail a été présenté à la **18th International Conference on Agents and Artificial Intelligence (ICAART 2026)** :

> Konstandinos Aiwansedo, Brahim Daoud et Hamza Ariouich. *Context-Aware AI-Based Framework for Sports Analytics*. Proceedings of the 18th International Conference on Agents and Artificial Intelligence – Volume 4: ICAART, SciTePress, 2026, pp. 3736–3743. DOI : [10.5220/0014469700004052](https://doi.org/10.5220/0014469700004052)

Si vous utilisez ce projet dans vos travaux, merci de citer l'article :

```bibtex
@conference{icaart26,
  author={Konstandinos Aiwansedo and Brahim Daoud and Hamza Ariouich},
  title={Context-Aware AI-Based Framework for Sports Analytics},
  booktitle={Proceedings of the 18th International Conference on Agents and Artificial Intelligence - Volume 4: ICAART},
  year={2026},
  pages={3736-3743},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0014469700004052},
  isbn={978-989-758-796-2},
}
```

## 📄 Licence

Copyright © 2026 Konstandinos Aiwansedo, Brahim Daoud, Hamza Ariouich.

Ce projet est distribué sous licence **GNU Affero General Public License v3.0 (AGPL-3.0)**. Voir le fichier [LICENSE](LICENSE).

En résumé :
- ✅ vous pouvez utiliser, étudier et modifier ce code ;
- 📝 toute redistribution ou version modifiée doit conserver la mention des auteurs et être publiée sous la même licence AGPL-3.0, avec son code source ;
- 🌐 cette obligation s'applique aussi si le logiciel est mis à disposition **via un réseau** (application web, API, SaaS) ;
- ❌ il est interdit d'intégrer ce code dans un logiciel propriétaire ou fermé.

Pour tout usage commercial hors de ce cadre (licence propriétaire), contactez les auteurs.
