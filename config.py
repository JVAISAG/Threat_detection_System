"""
Configuration file for the Distributed Security Threat Detection System
"""

# Federated Learning Configuration
FL_CONFIG = {
    'num_clients': 10,
    'num_rounds': 50,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.001,
    'client_fraction': 1.0,  # Fraction of clients to use per round
}

# Multimodal Model Configuration
MODEL_CONFIG = {
    'text_embedding_dim': 768,  # BERT embedding dimension
    'image_feature_dim': 512,   # CNN feature dimension
    'sensor_feature_dim': 128,  # Sensor data feature dimension
    'fusion_dim': 1024,         # Fused feature dimension
    'num_classes': 2,           # Binary classification: normal/threat
    'dropout_rate': 0.3,
}

# Data Configuration
DATA_CONFIG = {
    'train_ratio': 0.7,
    'test_ratio': 0.3,
    'multimodal_weights': {
        'text': 0.4,
        'image': 0.3,
        'sensor': 0.3,
    }
}

# Differential Privacy Configuration
PRIVACY_CONFIG = {
    'noise_stddev': 0.01,  # Standard deviation for Gaussian noise
    'enable_dp': True,     # Enable differential privacy
}

# System Configuration
SYSTEM_CONFIG = {
    'communication_protocol': 'HTTP/2',
    'use_gpu': True,
    'random_seed': 42,
}
