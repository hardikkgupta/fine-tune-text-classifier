
# Fine-Tuning BERT for Text Classification (SST-2 Dataset)

This project demonstrates the fine-tuning of BERT for binary sentiment classification using the Stanford Sentiment Treebank (SST-2) dataset. The model is trained using PyTorch and HuggingFace Transformers, and the training process is logged via WandB. We also provide error analysis for incorrect predictions and a PCA visualization of BERT embeddings.

## Features

- **Model**: BERT (HuggingFace's `bert-base-uncased`)
- **Dataset**: SST-2 (Stanford Sentiment Treebank)
- **Training**: Custom training loop with a modified HuggingFace `Trainer`
- **Evaluation**: Logs accuracy and loss metrics; stores incorrect predictions for further analysis
- **Error Analysis**: Logs incorrect predictions, their confidence scores, and potential error types
- **Embedding Visualization**: Uses PCA to visualize BERT embeddings of correct and incorrect predictions

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/hardikkgupta/fine-tune-text-classifier.git
cd fine-tune-text-classifier
```

2. Install the required dependencies:

```bash
pip install transformers datasets wandb
```

## Usage

### 1. Model Training

Run the `hero.ipynb` notebook to train the BERT model on the SST-2 dataset. The notebook contains a custom training loop and logs the results to WandB for tracking.

To resume training from a checkpoint, set the `resume_training` flag to `True`.

### 2. Inference

Use the `single_prediction` function to test the model on a new text input. Example:

```python
sample_text = "I absolutely adore studying natural language processing!"
result, latency = single_prediction(sample_text)
print(f"Model Prediction for '{sample_text}': {result}")
print(f"Inference Time: {latency:.2f} ms")
```

### 3. Error Analysis

After evaluation, incorrect predictions are logged into a CSV file (`incorrect_predictions.csv`). This file contains the text, actual labels, predicted labels, and confidence scores for the incorrect predictions.

### 4. Embedding Visualization

The PCA of BERT embeddings for correct and incorrect predictions is plotted using matplotlib. Correct positive, correct negative, and incorrect predictions are shown in different colors.

## Configuration

You can configure the WandB logging and training parameters by modifying the `wandb_config` dictionary:

```python
wandb_config = {
    'epochs': 6,
    'num_classes': 2,
    'batch_size': 128,
    'learning_rate': 2e-5,
    'dataset': 'SST-2',
    'model_architecture': 'BERT'
}
```

## License

This project is licensed under the MIT License.
