import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import pathlib

import pandas as pd
import numpy as np
from PIL import Image
from typing import List
import io

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, input_file_name, col, pandas_udf, PandasUDFType
from pyspark.sql.types import *

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

support_target = ["Disease_Risk", "DR", "MH"]
target_list = ["disease-risk", 'dr', 'mh']
target_description = [
    "disease-risk : Presence of disease/abnormality",
    "dr : Presence of diabetic retinopathy",
    "mh : Presence of media haze"
]

support_model = ["RN50_LR", "RN50_NB"]
model_list = ["rn50-lr", "rn50-nb"]
model_description = [
    "rn50-lr : ResNet50 as deep featurizer and Logistic Regression as classifier",
    "rn50-bn : ResNet50 as deep featurizer and Naive Bayes as classifier"
]

# create spark session #######################################################

spark = SparkSession.builder\
    .config("spark.driver.memory", "6g")\
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")\
    .appName("Retinal-Disease-Classification").getOrCreate()
sc = spark.sparkContext

# defined function for spark #################################################

def get_filename_from_origin(path):
    filename = path.split("/")[-1].split(".")
    return filename[0] if len(filename) < 2 else "".join(filename[:-1])

filenameUDF = udf(lambda path: get_filename_from_origin(path), StringType())

def get_extension_from_origin(path):
    return path.split("/")[-1].split(".")[-1]

extensionUDF = udf(lambda path: get_extension_from_origin(path), StringType())

# deep feature extraction #####################################################

model = ResNet50(include_top=False)
bc_model_weights = sc.broadcast(model.get_weights())

def model_fn():
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model

def preprocess(content, resize=(224, 224)):
    img = Image.open(io.BytesIO(content)).resize(resize)
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
        
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

def data_preprocessing(images_path: str, target_col: str="Disease_Risk", labels_path: str=None):
    if labels_path is not None:
        _label = spark.read.option("delimiter", ",").option("header", True)\
        .csv(labels_path)
    
    _images = spark.read.format('binaryFile')\
    .option("pathGlobFilter", "*.png")\
    .option("dropInvalid", True).load(images_path)\
    .withColumn('filename', filenameUDF(input_file_name()))\
    .withColumn('extension', extensionUDF(input_file_name()))\
    .select('filename', 'extension', 'content', 'length', 'path')
    
    if labels_path is not None:
        try:
            _df = _images.join(_label, _images.filename == _label.ID, "inner")\
            .select('ID', 'filename', 'extension', 'content', 'length', 'path', target_col)
        except:
            _images.printSchema()
            _label.printSchema()
            sys.exit()
    else:
        _df = _images
    
    if labels_path is not None:
        _feature = _df.repartition(16)\
                .select(
                    col("ID"), col('filename'),
                    col(target_col).alias('Target'),
                    featurize_udf("content").alias("features")
                    )
        _feature = _feature.select(
                    col("ID"), col('filename'), 
                    col('Target'),
                    list_to_vector_udf(_feature["features"]).alias("features"),
                    )
    else:
        _feature = _df.repartition(16)\
                .select(
                    col('filename'),
                    featurize_udf("content").alias("features")
                    )
        _feature = _feature.select(
                    col('filename'), 
                    list_to_vector_udf(_feature["features"]).alias("features"),
                    )
    
    return _feature

def load_model(model=support_model[0], target=support_target[0]):
    model_path = 'Model/' + model + '/' + target
    spark_model = PipelineModel.load(model_path)
    
    return spark_model

def predict(model, features):
    labels = model.stages[0].labels
    indexStringLayer = model.stages.pop(0)
    labelIndexReverser = IndexToString(inputCol="prediction", outputCol="predictLabel", labels=labels)
    predictions = model.transform(features)
    predictions = labelIndexReverser.transform(predictions)
    
    print("Prediction Result")
    if 'Target' in predictions.columns:
        predictions.select('ID', 'filename', 'Target', col('predictLabel').alias('Predict')).show()   
    else: 
        predictions.select('filename', col('predictLabel').alias('Predict')).show()

# main function ###############################################################

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s <IMAGE> [OPTION]",
        description="Classify retinal disease from image."
    )
    parser.add_argument(
        'image', help="path to an image of image's directory (support only PNG image)"
    )
    parser.add_argument(
        '-l','--label', help="path to an label csv file"
    )
    parser.add_argument(
        '-t', '--target', default=target_list[0],
        help="target to classify (default: disease-risk). support target; " +\
            ", ".join(target_description)
    )
    parser.add_argument(
        '-m', '--model', default=model_list[0],
        help="Classification model (default: rn50-lr). support model; " +\
            ", ".join(model_description)
    )
    return parser

def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    input_path = args.image
    label_path = args.label
    
    print(f'Image path: {pathlib.Path(input_path)}')
    if not os.path.isdir(input_path) and not os.path.isfile(input_path):
        print('The path or file specified does not exist')
        sys.exit()
    elif not input_path.endswith('.png') and os.path.isfile(input_path):
        print('Do not support file type, expected PNG')
        sys.exit()
    
    if label_path:
        print(f'Label file: {pathlib.Path(label_path)}')
        if not os.path.isfile(label_path):
            print('The path or file specified does not exist')
            sys.exit()
        elif not label_path.endswith('.csv'):
            print('Do not support file type, expected CSV')
            sys.exit()
    
    if args.target not in target_list:
        print('Do not support specified target;', args.target)
        sys.exit()
    
    if args.model not in model_list:
        print('Do not support specified model;', args.model)
        sys.exit()
    
    target_indx = target_list.index(args.target)
    target = support_target[target_indx]
    print(f"Target: {target}")
    
    model_indx = model_list.index(args.model)
    model = support_model[model_indx]
    print(f"Classification Model: {model}")
    
    features = data_preprocessing(input_path, target, labels_path=label_path)
    classifier = load_model(model, target)
    predict(classifier, features)
    
        
if __name__ == "__main__":
    main()