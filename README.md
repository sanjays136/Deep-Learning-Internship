
# Internship Assignment on Deep Learning

This repository contains the solution to the assignment on Feedforward Neural Network that was completed as part of enhancing the learning experience in the Deep Learning internship offered by the Indian Institute of Technology Madras, taught by Prof. Mitesh Khapra.

## Libraries Used

- `copy`: To obtain a deep copy of the class Model.
- `tqdm`: To track the time left in a particular run.
- `wandb`: To log all the runs with their metrics.
- `matplotlib` and `seaborn`: To plot graphs such as confusion matrix and ROC curves.

## Instructions

### Dependencies

In addition, install the following packages (or go pip install -r requirements.txt):
- Python 3.7+
- Keras

  
### Installation

Install the required packages using the following command:
pip install numpy tqdm wandb matplotlib copy argparse keras


Running the Code
To train and evaluate the Feedforward Neural Network, follow these steps:

1. Clone the repository:


2. Run the training script:

python train_model.py 


3. Evaluate the model:

python evaluate_model.py


## Results

The model performance and metrics will be logged using wandb. Graphs such as confusion matrix and ROC curves will be generated using matplotlib and seaborn.

## Acknowledgments

This project was developed under the guidance of Prof. Mitesh Khapra as part of the Deep Learning Internship at the Indian Institute of Technology Madras.

