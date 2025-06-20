# Streaming-Fraud-Detection
## 📌 Description du Projet
Ce projet vise à développer un système de détection de fraudes en temps réel en utilisant des technologies Big Data et des modèles de Machine Learning. L'objectif est d'analyser des flux de transactions financières pour identifier les comportements suspects et générer des alertes immédiates. Le système intègre des outils comme Apache Kafka, Apache Spark, Cassandra, Docker et Streamlit pour assurer une solution scalable et performante.

---

## 🚀 Fonctionnalités Clés
- **Pipeline de données complet** : Ingestion, traitement et stockage des transactions.
- **Détection en temps réel** : Analyse des transactions en quelques millisecondes.
- **Multi-modèles** : Utilisation de plusieurs algorithmes de Machine Learning (Random Forest, Régression Logistique, GBT).
- **Système d'alerte** : Notifications par email en cas de fraude détectée.
- **Visualisation interactive** : Tableaux de bord Power BI et interface Streamlit pour le suivi des fraudes.
- **Scalabilité** : Architecture conçue pour gérer des volumes élevés de données.

---

## 📊 Résultats et Performances
Les modèles de Machine Learning ont démontré des performances élevées :
- **Random Forest** : AUC = 0.987, Accuracy = 0.979, F1-score = 0.986.
- **Gradient Boosted Trees** : AUC = 0.969, Accuracy = 0.976, F1-score = 0.984.
- **Régression Logistique** : AUC = 0.864, Accuracy = 0.946, F1-score = 0.967.

L'analyse exploratoire a révélé des tendances clés :
- Les fraudes sont plus fréquentes entre **200$ et 1000$** et surviennent **majoritairement autour de minuit**.
- Les catégories **shopping_net** et **misc_net** sont les plus vulnérables.
- Les pics de fraude sont observés en **janvier, novembre et décembre**.

---

## 🛠️ Technologies Utilisées
- **Big Data** : Apache Kafka (flux de données), Apache Spark (traitement distribué), Cassandra (stockage NoSQL).
- **Machine Learning** : Random Forest, Régression Logistique, Gradient Boosted Trees.
- **Déploiement** : Docker (conteneurisation), Docker Compose (orchestration).
- **Visualisation** : Power BI (tableaux de bord), Streamlit (interface web interactive).
- **Autres** : Python, SMTP (envoi d'emails).

---
