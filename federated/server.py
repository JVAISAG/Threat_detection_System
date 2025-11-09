"""
Federated Learning Server for Model Aggregation
Implements FedAvg algorithm as described in the paper
"""

import torch
import numpy as np
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FL_CONFIG


class FederatedServer:
    """
    Federated Learning Server implementing FedAvg algorithm
    Implements equation (1): θ_global = 1/N * Σ θ_i
    """
    def __init__(self, global_model, device='cpu'):
        self.global_model = global_model
        self.device = device
        self.global_model.to(device)

        # Track training history
        self.round_history = []

    def aggregate_parameters(self, client_parameters_list, client_weights=None):
        """
        Aggregate client model parameters using FedAvg
        Implements equation (1) from the paper

        Args:
            client_parameters_list: List of parameter dictionaries from clients
            client_weights: Optional weights for weighted averaging (based on dataset size)

        Returns:
            Aggregated global parameters
        """
        if len(client_parameters_list) == 0:
            raise ValueError("No client parameters to aggregate")

        # If no weights provided, use equal weights
        if client_weights is None:
            client_weights = [1.0 / len(client_parameters_list)] * len(client_parameters_list)
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated parameters with zeros
        global_parameters = {}
        first_params = client_parameters_list[0]

        for key in first_params.keys():
            global_parameters[key] = torch.zeros_like(first_params[key])

        # Weighted average of client parameters
        # θ_global = Σ (w_i * θ_i)
        for client_params, weight in zip(client_parameters_list, client_weights):
            for key in client_params.keys():
                global_parameters[key] += weight * client_params[key]

        return global_parameters

    def update_global_model(self, client_parameters_list, client_weights=None):
        """
        Update global model with aggregated parameters
        """
        aggregated_params = self.aggregate_parameters(
            client_parameters_list,
            client_weights
        )
        self.global_model.set_parameters(aggregated_params)

    def get_global_parameters(self):
        """
        Get current global model parameters to send to clients
        """
        return self.global_model.get_parameters()

    def federated_learning_round(self, clients, round_num):
        """
        Execute one round of federated learning

        Args:
            clients: List of FederatedClient objects
            round_num: Current round number

        Returns:
            Dictionary with round statistics
        """
        print(f"\n=== Federated Learning Round {round_num} ===")

        # Select clients for this round
        num_clients = max(1, int(len(clients) * FL_CONFIG['client_fraction']))
        selected_clients = np.random.choice(clients, num_clients, replace=False)

        # Distribute global model to selected clients
        global_params = self.get_global_parameters()
        for client in selected_clients:
            client.set_parameters(deepcopy(global_params))

        # Local training on each client
        client_parameters = []
        client_losses = []
        client_weights = []

        for client in selected_clients:
            print(f"Training Client {client.client_id}...")
            loss = client.train_local()
            params = client.get_parameters()

            client_parameters.append(params)
            client_losses.append(loss)
            client_weights.append(len(client.train_data))

            print(f"Client {client.client_id} - Loss: {loss:.4f}")

        # Aggregate client updates
        self.update_global_model(client_parameters, client_weights)

        # Calculate average loss
        avg_loss = np.mean(client_losses)

        round_stats = {
            'round': round_num,
            'avg_loss': avg_loss,
            'num_clients': len(selected_clients),
        }

        self.round_history.append(round_stats)

        print(f"Round {round_num} - Average Loss: {avg_loss:.4f}")

        return round_stats

    def train(self, clients, num_rounds=None):
        """
        Run federated learning for multiple rounds

        Args:
            clients: List of FederatedClient objects
            num_rounds: Number of federated learning rounds

        Returns:
            Training history
        """
        if num_rounds is None:
            num_rounds = FL_CONFIG['num_rounds']

        print(f"Starting Federated Learning with {len(clients)} clients for {num_rounds} rounds")

        for round_num in range(1, num_rounds + 1):
            self.federated_learning_round(clients, round_num)

        print("\nFederated Learning Complete!")
        return self.round_history

    def evaluate_global_model(self, test_data_loader, device=None):
        """
        Evaluate the global model on test data

        Args:
            test_data_loader: DataLoader with test data
            device: Device to run evaluation on

        Returns:
            Evaluation metrics
        """
        if device is None:
            device = self.device

        self.global_model.eval()
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        with torch.no_grad():
            for batch in test_data_loader:
                # Unpack batch
                text_ids = batch.get('text_input_ids', None)
                text_mask = batch.get('text_attention_mask', None)
                images = batch.get('images', None)
                sensor_data = batch.get('sensor_data', None)
                labels = batch['labels']

                # Move to device
                if text_ids is not None:
                    text_ids = text_ids.to(device)
                    text_mask = text_mask.to(device)
                if images is not None:
                    images = images.to(device)
                if sensor_data is not None:
                    sensor_data = sensor_data.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.global_model(
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

        metrics = {
            'accuracy': accuracy,
            'false_positive_rate': fpr * 100,
            'false_negative_rate': fnr * 100,
            'total_samples': total,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
        }

        return metrics
