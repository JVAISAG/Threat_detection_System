"""
Multimodal LLM for processing heterogeneous security data
Processes text (logs), images (device images), and sensor data
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG


class TextEncoder(nn.Module):
    """BERT-based text encoder for processing security logs"""
    def __init__(self, embedding_dim=768):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)


class ImageEncoder(nn.Module):
    """CNN-based image encoder for processing device/network images"""
    def __init__(self, feature_dim=512):
        super(ImageEncoder, self).__init__()
        # Use pretrained ResNet as backbone
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, images):
        features = self.features(images)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class SensorEncoder(nn.Module):
    """MLP-based encoder for processing sensor/network traffic data"""
    def __init__(self, input_dim=100, feature_dim=128):
        super(SensorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(self, sensor_data):
        return self.encoder(sensor_data)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal LLM that fuses text, image, and sensor data for threat detection
    Implements weighted fusion as described in equation (2) of the paper
    """
    def __init__(self, config=MODEL_CONFIG):
        super(MultimodalFusionModel, self).__init__()

        # Encoders for each modality
        self.text_encoder = TextEncoder(config['text_embedding_dim'])
        self.image_encoder = ImageEncoder(config['image_feature_dim'])
        self.sensor_encoder = SensorEncoder(feature_dim=config['sensor_feature_dim'])

        # Learnable weights for weighted fusion
        self.w_text = nn.Parameter(torch.tensor(0.4))
        self.w_image = nn.Parameter(torch.tensor(0.3))
        self.w_sensor = nn.Parameter(torch.tensor(0.3))

        # Projection layers to common dimension for each modality
        self.text_proj = nn.Linear(config['text_embedding_dim'], config['fusion_dim'])
        self.image_proj = nn.Linear(config['image_feature_dim'], config['fusion_dim'])
        self.sensor_proj = nn.Linear(config['sensor_feature_dim'], config['fusion_dim'])

        # Fusion layer (now works with fixed fusion_dim regardless of which modalities are present)
        self.fusion = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
        )

        # Classification head
        self.classifier = nn.Linear(config['fusion_dim'] // 2, config['num_classes'])

    def forward(self, text_input_ids=None, text_attention_mask=None,
                images=None, sensor_data=None):
        """
        Forward pass with multimodal fusion
        Implements X_fused = sum(w_i * X_i) from the paper
        """
        batch_size = (text_input_ids.size(0) if text_input_ids is not None
                     else images.size(0) if images is not None
                     else sensor_data.size(0))

        # Initialize fused features with zeros
        fused_features = torch.zeros(batch_size, self.fusion[0].in_features,
                                     device=next(self.parameters()).device)

        # Counter for available modalities
        modality_count = 0
        total_weight = 0.0

        # Encode and project text if available
        if text_input_ids is not None:
            text_features = self.text_encoder(text_input_ids, text_attention_mask)
            text_projected = self.text_proj(text_features)
            fused_features += self.w_text * text_projected
            total_weight += self.w_text
            modality_count += 1

        # Encode and project images if available
        if images is not None:
            image_features = self.image_encoder(images)
            image_projected = self.image_proj(image_features)
            fused_features += self.w_image * image_projected
            total_weight += self.w_image
            modality_count += 1

        # Encode and project sensor data if available
        if sensor_data is not None:
            sensor_features = self.sensor_encoder(sensor_data)
            sensor_projected = self.sensor_proj(sensor_features)
            fused_features += self.w_sensor * sensor_projected
            total_weight += self.w_sensor
            modality_count += 1

        # Normalize by total weight (implements weighted sum from equation 2)
        if modality_count > 0 and total_weight > 0:
            fused_features = fused_features / total_weight

        # Apply fusion layers
        fused = self.fusion(fused_features)

        # Classification
        logits = self.classifier(fused)

        return logits

    def get_parameters(self):
        """Get model parameters for federated learning"""
        return self.state_dict()

    def set_parameters(self, parameters):
        """Set model parameters from federated aggregation"""
        self.load_state_dict(parameters)
