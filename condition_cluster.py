import argparse
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=0)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(file,header=None) for file in input_files ]
    train_data = pd.concat(raw_data)
    print(train_data.shape)
    normalizer = Normalizer(copy=False)
    normalized_data = normalizer.fit_transform(train_data)
    print(normalized_data.shape)
    kmeans = KMeans(n_clusters=args.n_clusters,random_state=args.random_state).fit(normalized_data)
    
    
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(kmeans, os.path.join(args.model_dir, "kmeansmodel.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    kmeans = joblib.load(os.path.join(model_dir, "kmeansmodel.joblib"))
    return kmeans