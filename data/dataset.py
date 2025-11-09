"""
Multimodal Security Dataset
Handles text (logs), image (device images), and sensor (network traffic) data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms


class MultimodalSecurityDataset(Dataset):
    """
    Dataset for multimodal security threat detection
    Combines text logs, images, and sensor data
    """
    def __init__(self, data_samples, tokenizer=None, transform=None, max_length=128):
        """
        Args:
            data_samples: List of dictionaries with keys:
                - 'text': Security log text (optional)
                - 'image': Image path or PIL Image (optional)
                - 'sensor': Sensor data array (optional)
                - 'label': 0 for normal, 1 for threat
            tokenizer: BERT tokenizer for text processing
            transform: Image transformations
            max_length: Maximum sequence length for text
        """
        self.data_samples = data_samples
        self.max_length = max_length

        # Initialize tokenizer for text
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        # Initialize image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        result = {}

        # Process text data (security logs)
        if 'text' in sample and sample['text'] is not None:
            encoding = self.tokenizer(
                sample['text'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['text_input_ids'] = encoding['input_ids'].squeeze(0)
            result['text_attention_mask'] = encoding['attention_mask'].squeeze(0)

        # Process image data (device/network images)
        if 'image' in sample and sample['image'] is not None:
            if isinstance(sample['image'], str):
                image = Image.open(sample['image']).convert('RGB')
            else:
                image = sample['image']
            result['images'] = self.transform(image)

        # Process sensor data (network traffic features)
        if 'sensor' in sample and sample['sensor'] is not None:
            result['sensor_data'] = torch.tensor(
                sample['sensor'],
                dtype=torch.float32
            )

        # Add label
        result['labels'] = torch.tensor(sample['label'], dtype=torch.long)

        return result


def create_synthetic_data(num_samples=1000, threat_ratio=0.3):
    """
    Create synthetic multimodal security data for testing
    In production, this would load real security logs, device images, and network traffic

    Args:
        num_samples: Number of samples to generate
        threat_ratio: Ratio of threat samples (rest are normal)

    Returns:
        List of data samples
    """
    samples = []

    # Security log templates
    normal_logs = [
        "User login successful from IP 192.168.1.100",
        "System backup completed successfully",
        "Database connection established",
        "File access granted to authorized user",
        "Network traffic within normal parameters",
    ]

    threat_logs = [
        "Multiple failed login attempts detected from IP 10.0.0.50",
        "Unusual port scanning activity detected",
        "SQL injection attempt in web request",
        "Unauthorized access attempt to sensitive directory",
        "Abnormal network traffic pattern detected",
        "Potential DDoS attack: high volume of requests",
    ]

    for i in range(num_samples):
        is_threat = np.random.random() < threat_ratio

        sample = {
            'label': 1 if is_threat else 0
        }

        # Generate text (security log)
        if is_threat:
            sample['text'] = np.random.choice(threat_logs)
        else:
            sample['text'] = np.random.choice(normal_logs)

        # Generate synthetic sensor data (network traffic features)
        # Features: packet_rate, byte_rate, connection_count, port_diversity, etc.
        if is_threat:
            # Anomalous traffic patterns
            sample['sensor'] = np.random.randn(100) + np.array([2.0] * 100)
        else:
            # Normal traffic patterns
            sample['sensor'] = np.random.randn(100)

        # For synthetic data, we'll skip images to keep it simple
        # In production, you would include device/network images
        sample['image'] = None

        samples.append(sample)

    return samples


def split_data_for_clients(data_samples, num_clients=10):
    """
    Split data into non-IID partitions for federated learning clients
    Each client gets a subset of data

    Args:
        data_samples: List of data samples
        num_clients: Number of federated clients

    Returns:
        List of data subsets, one for each client
    """
    np.random.shuffle(data_samples)

    # Split data into chunks (non-IID distribution)
    client_data = []
    samples_per_client = len(data_samples) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(data_samples)
        client_data.append(data_samples[start_idx:end_idx])

    return client_data


class ThreatDetectionDataModule:
    """
    Data module for managing training and test datasets
    """
    def __init__(self, num_samples=10000, num_clients=10, threat_ratio=0.3):
        self.num_samples = num_samples
        self.num_clients = num_clients
        self.threat_ratio = threat_ratio

        # Generate synthetic data
        print("Generating synthetic multimodal security data...")
        all_data = create_synthetic_data(num_samples, threat_ratio)

        # Split into train and test
        split_idx = int(0.7 * len(all_data))
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]

        # Split training data for clients
        self.client_datasets = [
            MultimodalSecurityDataset(client_data)
            for client_data in split_data_for_clients(train_data, num_clients)
        ]

        # Test dataset
        self.test_dataset = MultimodalSecurityDataset(test_data)

        print(f"Created {num_clients} client datasets")
        print(f"Training samples per client: ~{len(train_data)//num_clients}")
        print(f"Test samples: {len(test_data)}")

    def get_client_dataset(self, client_id):
        """Get dataset for a specific client"""
        return self.client_datasets[client_id]

    def get_test_dataset(self):
        """Get test dataset"""
        return self.test_dataset
