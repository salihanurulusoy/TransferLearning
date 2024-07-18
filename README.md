...
├── TrainSet/
│ ├── class1/
│ └── class2/
├── ValidationSet/
│ ├── class1/
│ └── class2/
└── TestSet/
├── class1/
└── class2/

Glaucoma Detection with Transfer Learning Models
This project represents a study aimed at glaucoma detection using visual data analysis and transfer learning techniques.

Data Set
ORIGA Data Set: The ORIGA dataset contains retinal images for glaucoma diagnosis. It is organized in three folders: TrainSet, TestSet and ValidationSet.

Models Used
InceptionV3: The InceptionV3 model trained on the ImageNet dataset was used for transfer learning.
VGG19: VGG19 architecture is used for transfer learning for feature extraction and classification.
AlexNet: AlexNet is suitable for transfer learning experiments on glaucoma detection with a lighter architecture.

Installation
Requirements: Python 3.x, TensorFlow, Keras, scikit-learn, pandas, numpy and other libraries installed.

Execution: The code can be run by updating the train_dir, test_dir, validation_dir variables according to the directories where the dataset is located.

Training and Evaluation
Separate training and evaluation steps were performed for each model. During training, callback functions such as early stopping and model checking were used.

Results
A grid search was performed on the validation data to identify the best model. Performance metrics such as accuracy, precision, recall and F1-score were calculated on the test dataset.
