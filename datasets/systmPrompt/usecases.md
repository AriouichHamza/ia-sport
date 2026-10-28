### UC-IP : Prédiction de blessures (Injury Prediction)

**ID**  
UC-IP

**Titre**  
Prédiction de blessures

**Acteur principal**  
Entraîneur / Staff médical

**Objectif**  
Évaluer chaque jour la probabilité qu’un athlète se blesse lors de la séance du lendemain.

**Pré-conditions**  
- L’athlète est enregistré dans le système.  
- Historique complet de l’entraînement, des charges supportées et des blessures passées disponible.  
- Données de performance (séries temporelles) importées et normalisées.

**Déclencheur**  
Exécution quotidienne automatique du modèle la veille de chaque entraînement ou compétition.

**Scénario principal**  
1. Le système récupère les données historiques de l’athlète (charge, performances, blessures).  
2. Le modèle IP calcule, sur la base de ces séries temporelles, la probabilité de blessure.  
3. Le système génère une alerte si le risque dépasse un seuil prédéfini.  
4. L’entraîneur/staff reçoit le rapport de probabilité et recommandations préventives.

**Extensions**  
- 2a. Données manquantes ou incohérentes → Le système demande un complément ou signale l’erreur.  
- 3a. Risque élevé → Proposition automatique d’allègement de charge ou de séance alternative.  
- 4a. Suivi post-alerte → Enregistrement des actions correctives et retour d’expérience.

**Post-conditions**  
- Rapport de risque stocké en base.  
- Alerte transmise à l’équipe de suivi.
- Plan d’entraînement éventuellement ajusté.

**Notes techniques**  
- Modèle agnostique au sport, entraîné sur des données déséquilibrées (techniques de rééchantillonnage ou perte pondérée).  
- Pipeline de séries temporelles (windowing, normalisation, features de tendance).  
- Déploiement en batch quotidien ou via API interne.

---

### UC-RPE : Prédiction de la perception de l’effort (RPE Prediction)

**ID**  
UC-RPE

**Titre**  
Prédiction du Rate of Perceived Exertion

**Acteur principal**  
Entraîneur / Coach

**Objectif**  
Anticiper l’échelle de difficulté perçue (1–10) par l’athlète avant la séance.

**Pré-conditions**  
- Données objectives disponibles : charge d’entraînement, fréquence cardiaque, sommeil, récupération.  
- Historique de RPE fourni par l’athlète.

**Déclencheur**  
Demande de prédiction avant chaque séance ou planification hebdomadaire.

**Scénario principal**  
1. Le système compile les données objectives récentes et l’historique de RPE.  
2. Le modèle IA prédit le RPE attendu pour la prochaine séance.  
3. Le système affiche le score RPE et propose un ajustement de la charge si nécessaire.  
4. Le coach valide ou modifie la séance, en tenant compte de la prédiction.

**Extensions**  
- 1a. Données insuffisantes → Demande d’enrichissement (questionnaire, capteurs manquants).  
- 3a. RPE prédite > seuil → Suggestion automatique de réduction de charge ou repos.  
- 4a. Coach rejette la suggestion → Enregistrement de la décision et des motifs.

**Post-conditions**  
- Prédiction de RPE et recommandations archivées.  
- Séance ajustée ou confirmée.

**Notes techniques**  
- Features dérivées : tendance de la FC, qualité du sommeil, ratio charge/récupération.  
- Modèle supervisé (régression) ou réseaux neuronaux séquentiels (LSTM).  
- Interface de visualisation des tendances RPE.

---

### UC-PRM : Prédiction des résultats de matchs

**ID**  
UC-PRM

**Titre**  
Prédiction des résultats de matchs

**Acteur principal**  
Analyste / Entraîneur

**Objectif**  
Estimer la probabilité de victoire, nul ou défaite avant un match.

**Pré-conditions**  
- Historique détaillé des performances des équipes et des joueurs.  
- Statistiques de confrontations, conditions de jeu et forme du moment importées.

**Déclencheur**  
Exécution chaque veille de match ou à la demande.

**Scénario principal**  
1. Le système agrège les indicateurs historiques et contextuels.  
2. Le modèle génère les probabilités (victoire/nul/défaite).  
3. Le rapport est envoyé à l’analyste avec visualisation des facteurs clés.  
4. L’entraîneur ajuste la stratégie ou la composition d’équipe si nécessaire.

**Extensions**  
- 1a. Données manquantes (joueur blessé, météo) → Complément manuel ou estimation par défaut.  
- 2a. Confiance faible du modèle → Proposition de collecte de données supplémentaires.  
- 4a. Entraîneur rejette l’analyse → Enregistrement du feedback pour ré-entraînement futur.

**Post-conditions**  
- Rapport de probabilité archivé.  
- Décisions de stratégie consignées.

**Notes techniques**  
- Modèle agnostique au sport, entraîné sur données déséquilibrées.  
- Utilisation de techniques d’explicabilité (SHAP, LIME) pour identiﬁer les drivers de la prédiction.  
- API REST pour intégration dans le portail interne.

---

### UC-REC : Recrutement de joueurs

**ID**  
UC-REC

**Titre**  
Analyse et sélection de joueurs pour recrutement

**Acteur principal**  
Directeur sportif / Scout

**Objectif**  
Identifier et évaluer automatiquement des profils de joueurs répondant aux besoins de l’équipe.

**Pré-conditions**  
- Base de données joueurs à jour (performances, âge, blessures, progression).  
- Critères de style de jeu et poste définis.

**Déclencheur**  
Lancement d’une campagne de recrutement ou requête ad hoc (poste, style).

**Scénario principal**  
1. Le système filtre les joueurs selon critères techniques et physiques.  
2. Calcul de scores de compatibilité (style, potentiel, risque de blessure).  
3. Présentation d’un tableau de profils classés.  
4. Le directeur sportif sélectionne les candidats pour approfondir l’analyse.

**Extensions**  
- 1a. Critères flous ou contradictoires → Proposition de pondération ou clarification.  
- 2a. Données manquantes (psychologie, terrain) → Enquête ou évaluation terrain.  
- 4a. Feedback du scout → Ajustement des pondérations de modèle.

**Post-conditions**  
- Listes de potentiels recrues éditables.  
- Scores et justifications stockés.

**Notes techniques**  
- Multi-critère (performance, âge, historique blessures, intangibles).  
- Utilisation de clustering et scoring personnalisé.  
- Dashboard interactif pour comparaison (« compare ces deux ailiers », « profil U21 »).

