import pyspark.sql.types as T
import pyspark.sql.functions as F
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler)

from constant import ROOT_PATH

def train(dataset):
    spark = SparkSession\
        .builder\
        .master('local[*]')\
        .appName("train_model") \
        .config("spark.jars", "/home/u22/spark-streaming-kafka/jars/synapseml_2.12-1.0.2.jar")\
        .config("spark.executor.extraClassPath", "/home/u22/spark-streaming-kafka/jars/synapseml_2.12-1.0.2.jar")\
        .config("spark.driver.extraClassPath", "/home/u22/spark-streaming-kafka/jars/synapseml_2.12-1.0.2.jar")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    df = spark.read.csv(dataset, header=True, inferSchema=True)
    df = df.withColumn("Class", F.col("Class").astype(T.IntegerType()))

    fraud_df = df.filter(F.col("Class") == 1).limit(102)
    non_fraud_df = df.filter(F.col("Class") == 0).limit(102)
    train_df = fraud_df.union(non_fraud_df)

    numerical_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                      'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
                      'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                      'V26', 'V27']
    
    categorical_cols = ['Time']
    label_column_name = 'Class'

    # Convert string into numerical indices
    index_cols = [c + 'Index' for c in categorical_cols]
    string_indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=index_cols,
        stringOrderType='alphabetAsc',
        handleInvalid='keep'
    )

    # One-hot encoding to integer indexes
    ohe_cols = [c + 'OHE' for c in categorical_cols]
    one_hot_encoder = OneHotEncoder(
        inputCols=index_cols,
        outputCols=ohe_cols,
        handleInvalid='error',
        dropLast=False
    )

    # Missing value imputation
    imputed_cols = [c + '_imputed' for c in numerical_cols]
    imputer = Imputer(
        strategy='median',
        inputCols=numerical_cols,
        outputCols=imputed_cols
    )

    # Combine numerical columns into single vector
    vec_assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol='numerical_features'
    )

    # Scale numerical features
    standard_scaler = StandardScaler(
        inputCol='numerical_features',
        outputCol='numerical_features_scaled'
    )

    # Combine one-hot encoded and scaled numerical features
    assembler_cols = ohe_cols + ['numerical_features_scaled']
    vec_assembler2 = VectorAssembler(
        inputCols=assembler_cols,
        outputCol='features'
    )

    # LightGBM classifier
    lgb_classifier = LightGBMClassifier(
        featuresCol='features',
        labelCol=label_column_name,
        verbosity=1  # Add verbosity to see more info
    )

    # Create pipeline with a LIST of stages
    pipeline = Pipeline(stages=[
        string_indexer,
        one_hot_encoder,
        imputer,
        vec_assembler,
        standard_scaler,
        vec_assembler2,
        lgb_classifier
    ])

    model = pipeline.fit(train_df)
    model.write().overwrite().save(f"{ROOT_PATH}/model")

    # Test
    pipelineModel = PipelineModel.load(f"{ROOT_PATH}/model")
    df = pipelineModel.transform(train_df)
    df.show()


if __name__ == '__main__':
    train(f"{ROOT_PATH}/creditcard.csv")