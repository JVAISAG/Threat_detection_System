"""
Federated Learning Client Node
Each client trains locally and uploads model updates to the server
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FL_CONFIG, PRIVACY_CONFIG


class FederatedClient:
    """
    Federated Learning Client implementing local training
    Implements equations (3) and (4) from the paper
    """
    def __init__(self, client_id, model, train_data, device='cpu'):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.device = device
        self.model.to(device)

        # Optimizer with dynamic learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=FL_CONFIG['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs=None):
        """
        Local training on client data
        Implements local loss L_i(θ_i) and gradient descent
        """
        if epochs is None:
            epochs = FL_CONFIG['local_epochs']

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        train_loader = DataLoader(
            self.train_data,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=True
        )

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch data
                text_ids = batch.get('text_input_ids', None)
                text_mask = batch.get('text_attention_mask', None)
                images = batch.get('images', None)
                sensor_data = batch.get('sensor_data', None)
                labels = batch['labels']

                # Move to device
                if text_ids is not None:
                    text_ids = text_ids.to(self.device)
                    text_mask = text_mask.to(self.device)
                if images is not None:
                    images = images.to(self.device)
                if sensor_data is not None:
                    sensor_data = sensor_data.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    images=images,
                    sensor_data=sensor_data
                )

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                # Implements θ_{i+1} = θ_i - α_i * ∇L_i(θ_i)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / (epochs * num_batches) if num_batches > 0 else 0

        return avg_loss

    def get_parameters(self):
        """
        Get local model parameters for aggregation
        Implements θ_i extraction
        """
        parameters = self.model.get_parameters()

        # Apply differential privacy if enabled
        if PRIVACY_CONFIG['enable_dp']:
            parameters = self._add_differential_privacy(parameters)

        return parameters

    def set_parameters(self, parameters):
        """
        Update local model with global parameters
        Implements θ_i = θ_global
        """
        self.model.set_parameters(parameters)

    def _add_differential_privacy(self, parameters):
        """
        Add differential privacy noise to model parameters
        Implements equation (5): θ̂_i = θ_i + N(0, σ²)
        """
        noisy_parameters = {}
        sigma = PRIVACY_CONFIG['noise_stddev']

        for key, value in parameters.items():
            noise = torch.normal(
                mean=0.0,
                std=sigma,
                size=value.shape,
                device=value.device
            )
            noisy_parameters[key] = value + noise

        return noisy_parameters

    def evaluate(self, test_data):
        """
        Evaluate model on test data
        Returns accuracy, false positive rate, and false negative rate
        """
        self.model.eval()
        test_loader = DataLoader(
            test_data,
            batch_size=FL_CONFIG['batch_size'],
            shuffle=False
        )

        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                text_ids = batch.get('text_input_ids', None)
                text_mask = batch.get('text_attention_mask', None)
                images = batch.get('images', None)
                sensor_data = batch.get('sensor_data', None)
                labels = batch['labels']

                # Move to device
                if text_ids is not None:
                    text_ids = text_ids.to(self.device)
                    text_mask = text_mask.to(self.device)
                if images is not None:
                    images = images.to(self.device)
                if sensor_data is not None:
                    sensor_data = sensor_data.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    images=images,
                    sensor_data=sensor_data
                )

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate confusion matrix components
                for pred, label in zip(predicted, labels):
                    if label == 1:  # Actual threat
                        if pred == 1:
                            true_positives += 1
                        else:
                            false_negatives += 1
                    else:  # Actual normal
                        if pred == 1:
                            false_positives += 1
                        else:
                            true_negatives += 1

        accuracy = 100 * correct / total if total > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0

        return {
            'accuracy': accuracy,
            'false_positive_rate': fpr * 100,
            'false_negative_rate': fnr * 100,
            'total_samples': total
        }
