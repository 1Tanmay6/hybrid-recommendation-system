import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime
import os
import json


class PyTorchModelTester:
    """
    Comprehensive testing suite for pre-trained PyTorch models.
    Specifically designed for the NCF_CNN model.
    """

    def __init__(
        self,
        model_path: str = 'models/NCF_CNN.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize the model tester.

        Args:
            model_path: Path to the pre-trained model
            device: Device to run the model on ('cuda' or 'cpu')
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.batch_size = batch_size
        self.setup_logging()
        self.load_model(model_path)

    def setup_logging(self):
        """Configure logging to track testing process."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=f'logs/model_testing_{
                datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str):
        """
        Load the pre-trained PyTorch model.

        Args:
            model_path: Path to the model file
        """
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def prepare_dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """
        Create a DataLoader for the test data.

        Args:
            X: Input features
            y: Target labels
            batch_size: Batch size for the DataLoader

        Returns:
            DataLoader object
        """
        class TestDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        if batch_size is None:
            batch_size = self.batch_size

        dataset = TestDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def get_predictions(
        self,
        dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get model predictions for the test data.

        Args:
            dataloader: DataLoader containing test data

        Returns:
            Tuple of (true labels, predicted labels, prediction probabilities)
        """
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        self.model.eval()
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_y_true.append(y_batch.cpu())
            all_y_pred.append(predictions.cpu())
            all_y_prob.append(probabilities.cpu())

        return (
            torch.cat(all_y_true),
            torch.cat(all_y_pred),
            torch.cat(all_y_prob)
        )

    def evaluate_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: torch.Tensor
    ) -> Dict:
        """
        Calculate various classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary containing all metrics
        """
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_prob = y_prob.numpy()

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_prob, multi_class='ovr')
        }

        # Add per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['per_class'] = {
            f'class_{k}': v
            for k, v in class_report.items()
            if k.isdigit()
        }

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        save_path: str
    ):
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        save_path: str
    ):
        """
        Plot ROC curves for each class.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Path to save the plot
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        y_true = y_true.numpy()
        y_prob = y_prob.numpy()
        n_classes = y_prob.shape[1]

        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def analyze_errors(
        self,
        X: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict:
        """
        Analyze prediction errors to identify patterns.

        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing error analysis
        """
        errors = y_true != y_pred
        error_indices = torch.where(errors)[0]

        error_analysis = {
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.sum() / len(y_true)),
            'error_indices': error_indices.tolist(),
            'misclassification_pairs': []
        }

        # Analyze misclassification patterns
        for idx in error_indices:
            error_analysis['misclassification_pairs'].append({
                'index': int(idx),
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx])
            })

        return error_analysis

    def run_full_test_suite(
        self,
        test_data: Tuple[torch.Tensor, torch.Tensor],
        output_dir: str = 'test_results'
    ) -> Dict:
        """
        Run complete test suite including all metrics and visualizations.

        Args:
            test_data: Tuple of (X_test, y_test)
            output_dir: Directory to save results

        Returns:
            Dictionary containing all test results
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info("Starting full test suite...")

        X_test, y_test = test_data
        dataloader = self.prepare_dataloader(X_test, y_test)

        # Get predictions
        y_true, y_pred, y_prob = self.get_predictions(dataloader)

        # Calculate metrics
        results = {
            'metrics': self.evaluate_metrics(y_true, y_pred, y_prob),
            'error_analysis': self.analyze_errors(X_test, y_true, y_pred)
        }

        # Generate visualizations
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )

        self.plot_roc_curve(
            y_true, y_prob,
            save_path=os.path.join(output_dir, 'roc_curves.png')
        )

        # Save results to JSON
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        self.logger.info("Test suite completed successfully!")
        return results


# Example usage
if __name__ == "__main__":
    # Create test data (replace with your actual test data)
    # Adjust shape based on your model's input requirements
    X_test = torch.randn(1000, 3, 224, 224)
    # Adjust number of classes as needed
    y_test = torch.randint(0, 10, (1000,))

    # Initialize tester
    tester = PyTorchModelTester(
        model_path='models/NCF_CNN.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run full test suite
    results = tester.run_full_test_suite((X_test, y_test))

    # Print summary of results
    print("\nTest Results Summary:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"\nTotal Errors: {results['error_analysis']['total_errors']}")
    print(f"Error Rate: {results['error_analysis']['error_rate']:.4f}")
