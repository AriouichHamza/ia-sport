---
title: "Spécifications des Use Cases"
author: ""
date: "`r Sys.Date()`"
output: html_document
---

# UC-IP : Prédiction de blessures

**ID** : UC-IP

**Acteur principal** : Entraîneur / Staff médical

**Objectif** : Estimer quotidiennement la probabilité qu’un athlète se blesse lors de la séance suivante.

## Pré-conditions

- Athlète enregistré avec historique d’entraînements, charges et blessures.
- Séries temporelles de performances importées et normalisées.

## Déclencheur

Exécution automatique du modèle la veille de chaque séance ou compétition.

## Scénario principal

1. Import des données historiques (charges, performances, blessures).
2. Calcul de la probabilité de blessure via le modèle IP.
3. Génération d’une alerte si le risque dépasse un seuil.
4. Envoi au coach/staff d’un rapport avec recommandations préventives.

## Extensions

- **2a.** Données manquantes/incohérentes → demande de compléments ou signalement d’erreur.
- **3a.** Risque élevé → proposition d’allègement de charge ou de séance alternative.
- **4a.** Suivi post-alerte → enregistrement des actions correctives et retours d’expérience.

## Post-conditions

- Stockage du rapport de risque.
- Alerte transmise à l’équipe.
- Ajustement éventuel du plan d’entraînement.

## Notes techniques

- Modèle agnostique au sport, gérant les déséquilibres (rééchantillonnage ou perte pondérée).
- Pipeline séries temporelles (fenêtrage, normalisation, extraction de tendances).
- Déploiement en batch quotidien ou via API interne.


# UC-RPE : Prédiction de la perception de l’effort

**ID** : UC-RPE

**Acteur principal** : Entraîneur / Coach

**Objectif** : Anticiper le Rate of Perceived Exertion (1–10) avant chaque séance.

## Pré-conditions

- Données objectives (charge, FC, sommeil, récupération).
- Historique des RPE fournies par l’athlète.

## Déclencheur

Demande avant chaque séance ou lors de la planification hebdomadaire.

## Scénario principal

1. Agrégation des données objectives récentes et de l’historique RPE.
2. Prédiction du RPE par le modèle IA.
3. Affichage du score RPE et suggestions d’ajustement de charge.
4. Validation ou modification de la séance par le coach.

## Extensions

- **1a.** Données insuffisantes → demande d’enrichissement (questionnaire, capteurs).
- **3a.** RPE prédite > seuil → suggestion de réduction de charge ou de repos.
- **4a.** Coach rejette → enregistrement de la décision et des motifs.

## Post-conditions

- Archivage de la prédiction et des recommandations.
- Séance ajustée ou confirmée.

## Notes techniques

- Features : tendance FC, qualité du sommeil, ratio charge/récupération.
- Modèles : régression supervisée ou LSTM.
- Interface de visualisation des tendances.


# UC-PRM : Prédiction des résultats de matchs

**ID** : UC-PRM

**Acteur principal** : Analyste / Entraîneur

**Objectif** : Estimer la probabilité de victoire, nul ou défaite avant un match.

## Pré-conditions

- Historique détaillé des équipes/joueurs.
- Statistiques de confrontations, conditions de jeu et forme actuelle.

## Déclencheur

Exécution la veille du match ou sur demande.

## Scénario principal

1. Agrégation des indicateurs historiques et contextuels.
2. Génération des probabilités (victoire/nul/défaite).
3. Envoi d’un rapport avec visualisation des facteurs clés.
4. Ajustement de la stratégie ou de la composition d’équipe.

## Extensions

- **1a.** Données manquantes (blessure, météo) → saisie manuelle ou estimation par défaut.
- **2a.** Confiance faible → proposition de collecte de données supplémentaires.
- **4a.** Entraîneur rejette → enregistrement du feedback pour réentraînement.

## Post-conditions

- Archivage du rapport.
- Consignation des décisions stratégiques.

## Notes techniques

- Modèle agnostique, entraîné sur données déséquilibrées.
- Explicabilité (SHAP, LIME) pour identifier les drivers.
- API REST pour intégration dans le portail interne.


# UC-REC : Recrutement de joueurs

**ID** : UC-REC

**Acteur principal** : Directeur sportif / Scout

**Objectif** : Identifier et évaluer automatiquement des profils adaptés aux besoins de l’équipe.

## Pré-conditions

- Base de données à jour (performances, âge, blessures, progression).
- Critères de style de jeu et postes définis.

## Déclencheur

Lancement d’une campagne de recrutement ou requête ad hoc.

## Scénario principal

1. Filtrage des joueurs selon critères techniques et physiques.
2. Calcul de scores de compatibilité (style, potentiel, risque).
3. Présentation d’un classement de profils.
4. Sélection des candidats pour analyse approfondie.

## Extensions

- **1a.** Critères flous → proposition de pondération ou clarification.
- **2a.** Données manquantes (psychologie, terrain) → enquête terrain.
- **4a.** Feedback du scout → ajustement des pondérations.

## Post-conditions

- Listes de recrues potentielles éditables.
- Scores et justifications stockés.

## Notes techniques

- Approche multi-critères (performance, âge, blessures, intangibles).
- Clustering et scoring personnalisé.
- Dashboard interactif (comparaison de profils, filtrage U21).
