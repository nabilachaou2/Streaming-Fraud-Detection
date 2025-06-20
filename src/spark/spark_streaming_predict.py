import logging
from pyspark.sql.functions import expr
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when, isnan, udf, lit, current_timestamp
from pyspark.sql.functions import current_date
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, LongType
from pyspark.sql.functions import udf
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql import DataFrame
import uuid
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialisation de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction pour envoyer l'alerte par email
def send_alert(transaction_id, amount, merchant, category, model_name):
    try:
        msg = MIMEMultipart()
        msg['From'] = "chakour.aya@etu.uae.ac.ma"
        msg['To'] = "nabilachaou321@gmail.com"
        msg['Subject'] = f"🚨 Fraud Alert: Transaction {transaction_id}"

        body = f"""
        🚨 Fraudulent Transaction Detected 🚨
        
        - ID: {transaction_id}
        - Amount: ${amount:.2f}
        - Merchant: {merchant}
        - Category: {category}
        - Model: {model_name}
        
        This transaction was flagged as potentially fraudulent.
        Please review it immediately.
        
        -- 
        Fraud Detection System
        """
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(msg['From'], "****************")  # Mot de passe d'application Gmail
            server.sendmail(msg['From'], msg['To'], msg.as_string())
            
        logger.info(f"✅ Alert sent for transaction {transaction_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send alert for transaction {transaction_id}: {str(e)}")
        return False

# Création de la session Spark avec configuration de mémoire
spark = SparkSession.builder \
    .appName("StreamingFraudDetection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Schéma pour les données de transaction
schema = StructType() \
    .add("trans_date_trans_time", StringType()) \
    .add("cc_num", StringType()) \
    .add("merchant", StringType()) \
    .add("category", StringType()) \
    .add("amt", DoubleType()) \
    .add("first", StringType()) \
    .add("last", StringType()) \
    .add("gender", StringType()) \
    .add("street", StringType()) \
    .add("city", StringType()) \
    .add("state", StringType()) \
    .add("zip", IntegerType()) \
    .add("lat", DoubleType()) \
    .add("long", DoubleType()) \
    .add("city_pop", IntegerType()) \
    .add("job", StringType()) \
    .add("dob", StringType()) \
    .add("trans_num", StringType()) \
    .add("unix_time", LongType()) \
    .add("merch_lat", DoubleType()) \
    .add("merch_long", DoubleType()) \
    .add("is_fraud", IntegerType(

# Liste des colonnes que tu veux conserver dans ta table Cassandra (avec nouvelle colonne Date)
selected_columns = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "transaction_id", "unix_time",
    "merch_lat", "merch_long", "prediction", "model"
]

# UDFs pour les alertes
@udf(returnType=StringType())
def generate_uuid():
    return str(uuid.uuid4())

@udf(returnType=StringType())
def determine_alert_type(prediction, amount):
    if prediction == 1:
        if amount > 1000:
            return "HIGH_VALUE_FRAUD"
        else:
            return "SUSPECTED_FRAUD"
    return None

@udf(returnType=DoubleType())
def calculate_risk_score(prediction, amount):
    if prediction == 1:
        base_score = 0.8
        # Ajoutez des facteurs supplémentaires au score si nécessaire
        amount_factor = min(0.2, amount / 10000)  # Max 0.2 supplémentaire basé sur montant
        return base_score + amount_factor
    return 0.0

# Lecture du flux de Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "transaction_data") \
    .option("startingOffsets", "latest") \
    .load()

# Parse du JSON
json_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Nettoyage des données et ajout de la colonne Date (timestamp actuel)
clean_df = json_df.na.fill(0) \
    .withColumn("amt", when(isnan("amt") | col("amt").isNull(), 0.0).otherwise(col("amt"))) \
    .withColumn("lat", when(isnan("lat") | col("lat").isNull(), 0.0).otherwise(col("lat"))) \
    .withColumn("long", when(isnan("long") | col("long").isNull(), 0.0).otherwise(col("long"))) \
    .withColumn("merch_lat", when(isnan("merch_lat") | col("merch_lat").isNull(), 0.0).otherwise(col("merch_lat"))) \
    .withColumn("merch_long", when(isnan("merch_long") | col("merch_long").isNull(), 0.0).otherwise(col("merch_long"))) 

# Dictionnaire des modèles à charger
models = {
    "GBT": "/home/aya/Streaming-Fraud-Detection/resultats/GBTClassifier_model"
}

# Fonction pour appliquer les modèles sur les lignes de données
def apply_models_to_rows(batch_df: DataFrame, batch_id: int):
    if batch_df.isEmpty():
        logger.info("Pas de données à traiter dans ce batch.")
        return

    # Ajouter la colonne transaction_id en amont avant d'appliquer les modèles
    current_batch = batch_df.withColumn("transaction_id", col("trans_num"))

    all_predictions = []
    fraud_predictions = []  # Pour collecter toutes les prédictions de fraude

    # Appliquer chaque modèle individuellement
    for model_name, model_path in models.items():
        try:
            logger.info(f"Chargement du modèle {model_name} depuis {model_path}")
            # Charger le modèle
            pipeline_model = PipelineModel.load(model_path)
            vector_assembler = pipeline_model.stages[0]
            required_cols = vector_assembler.getInputCols()
            
            # Ajouter les colonnes manquantes si nécessaire
            model_batch = current_batch
            for colname in required_cols:
                if colname not in model_batch.columns:
                    model_batch = model_batch.withColumn(colname, F.lit(0.0))
            
            # Appliquer le modèle
            logger.info(f"Application du modèle {model_name} sur les données.")
            pred_df = pipeline_model.transform(model_batch)
            
            # Ajouter le nom du modèle
            model_predictions = pred_df.withColumn("model", F.lit(model_name))
            
            # Filtrer pour ne conserver que les colonnes sélectionnées
            model_predictions = model_predictions.select(*selected_columns)
            
            # Collecter les prédictions frauduleuses pour les alertes
            fraud_df = model_predictions.filter(col("prediction") == 1)
            if not fraud_df.isEmpty():
                fraud_predictions.append(fraud_df)
                
                # Envoyer des emails pour chaque transaction frauduleuse détectée
                fraud_rows = fraud_df.collect()
                #for row in fraud_rows:  # FIXED: Uncommented the for loop
                    #send_alert(
                        #transaction_id=row["transaction_id"],
                        #amount=row["amt"],
                       # merchant=row["merchant"],
                     #   category=row["category"],
                      #  model_name=model_name )
            
            # Ajouter aux prédictions
            all_predictions.append(model_predictions)
            
            # Sauvegarder les prédictions du modèle dans Cassandra
            logger.info(f"Sauvegarde des prédictions du modèle {model_name} dans Cassandra.")
            model_predictions.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table="transactions_pred", keyspace="fraud_detection") \
                .save()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application du modèle {model_name}: {e}")

    # Traiter et enregistrer les alertes si des fraudes sont détectées
    if fraud_predictions and len(fraud_predictions) > 0:
        try:
            # Combine toutes les prédictions de fraude
            all_fraud_df = reduce(DataFrame.unionByName, fraud_predictions)
            logger.info(f"Génération d'alertes pour {all_fraud_df.count()} transactions frauduleuses détectées.")
            
            # Créer un DataFrame d'alertes
            alerts_df = all_fraud_df.select(
                generate_uuid().alias("alert_id"),
                col("transaction_id"),
                col("cc_num"),
                col("date").alias("alert_time"),  # Utiliser la nouvelle colonne Date
                col("amt"),
                col("merchant"),
                determine_alert_type(col("prediction"), col("amt")).alias("alert_type"),
                calculate_risk_score(col("prediction"), col("amt")).alias("risk_score"),
                lit("NEW").alias("status")
            )
            
            # Enregistrer les alertes dans Cassandra
            logger.info("Sauvegarde des alertes dans Cassandra.")
            alerts_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table="alerts", keyspace="fraud_detection") \
                .save()
            
            logger.info(f"Enregistré {alerts_df.count()} alertes dans Cassandra.")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des alertes: {e}")

    # Combiner toutes les prédictions si nécessaire
    if all_predictions and len(all_predictions) > 0:
        try:
            logger.info("Combinaison des prédictions des différents modèles.")
            # Union de toutes les prédictions
            final_predictions = reduce(DataFrame.unionByName, all_predictions)
        except Exception as e:
            logger.error(f"Erreur lors de la combinaison des prédictions: {e}")

# Démarrer la requête de streaming
logger.info("Démarrage du traitement en streaming.")
query = clean_df.writeStream \
    .foreachBatch(apply_models_to_rows) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/spark_checkpoint") \
    .start()

query.awaitTermination()