# GDP-HMM-Solution
A dose prediction method that placed sixth in the GDP-HMM Challenge.

## Data preprocessing
Place the training data in the folder named "train_data".

### Convert npz into nii

    python npz2nii.py

### Split data

    python split.py

## Model training

    python train.py

## Model inference
Place the test data in the folder named "data".

    python inference.py
