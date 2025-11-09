"""
Real-time Threat Detection System
Uses the trained federated multimodal model for threat detection
"""

import torch
import time
import numpy as np
from collections import deque


class ThreatDetector:
    """
    Real-time threat detection system using the trained model
    Provides fast inference and threat classification
    """
    def __init__(self, model, device='cpu', threshold=0.5):
        """
        Args:
            model: Trained multimodal fusion model
            device: Device to run inference on
            threshold: Confidence threshold for threat classification
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()

        # Statistics tracking
        self.detection_times = deque(maxlen=100)
        self.threat_count = 0
        self.total_detections = 0

    def detect_threat(self, text=None, image=None, sensor_data=None):
        """
        Detect threats in real-time from multimodal input

        Args:
            text: Security log text (optional)
            image: Device/network image (optional)
            sensor_data: Network traffic sensor data (optional)

        Returns:
            Dictionary with detection results
        """
        start_time = time.time()

        with torch.no_grad():
            # Prepare inputs
            batch_data = self._prepare_inputs(text, image, sensor_data)

            # Forward pass
            outputs = self.model(
                text_input_ids=batch_data.get('text_input_ids'),
                text_attention_mask=batch_data.get('text_attention_mask'),
                images=batch_data.get('images'),
                sensor_data=batch_data.get('sensor_data')
            )

            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            threat_prob = probabilities[0, 1].item()  # Probability of threat class
            predicted_class = 1 if threat_prob > self.threshold else 0

            # Calculate detection time
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)

            # Update statistics
            self.total_detections += 1
            if predicted_class == 1:
                self.threat_count += 1

            result = {
                'is_threat': bool(predicted_class == 1),
                'threat_probability': float(threat_prob),
                'normal_probability': float(probabilities[0, 0].item()),
                'confidence': float(max(probabilities[0]).item()),
                'detection_time_ms': detection_time * 1000,
                'threat_level': self._get_threat_level(threat_prob),
            }

        return result

    def _prepare_inputs(self, text, image, sensor_data):
        """
        Prepare inputs for the model
        """
        from transformers import BertTokenizer
        import torchvision.transforms as transforms
        from PIL import Image as PILImage

        batch_data = {}

        # Process text
        if text is not None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            batch_data['text_input_ids'] = encoding['input_ids'].to(self.device)
            batch_data['text_attention_mask'] = encoding['attention_mask'].to(self.device)

        # Process image
        if image is not None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            if isinstance(image, str):
                image = PILImage.open(image).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            batch_data['images'] = image_tensor

        # Process sensor data
        if sensor_data is not None:
            if isinstance(sensor_data, np.ndarray):
                sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
            sensor_tensor = sensor_data.unsqueeze(0).to(self.device)
            batch_data['sensor_data'] = sensor_tensor

        return batch_data

    def _get_threat_level(self, threat_prob):
        """
        Determine threat level based on probability
        """
        if threat_prob < 0.3:
            return "LOW"
        elif threat_prob < 0.6:
            return "MEDIUM"
        elif threat_prob < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def batch_detect(self, batch_samples):
        """
        Detect threats in a batch of samples

        Args:
            batch_samples: List of dictionaries with 'text', 'image', 'sensor_data'

        Returns:
            List of detection results
        """
        results = []
        for sample in batch_samples:
            result = self.detect_threat(
                text=sample.get('text'),
                image=sample.get('image'),
                sensor_data=sample.get('sensor_data')
            )
            results.append(result)
        return results

    def get_statistics(self):
        """
        Get detection statistics
        """
        avg_detection_time = np.mean(self.detection_times) * 1000 if self.detection_times else 0
        threat_rate = (self.threat_count / self.total_detections * 100) if self.total_detections > 0 else 0

        stats = {
            'total_detections': self.total_detections,
            'threats_detected': self.threat_count,
            'threat_rate_percent': threat_rate,
            'avg_detection_time_ms': avg_detection_time,
            'min_detection_time_ms': min(self.detection_times) * 1000 if self.detection_times else 0,
            'max_detection_time_ms': max(self.detection_times) * 1000 if self.detection_times else 0,
        }

        return stats

    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_times.clear()
        self.threat_count = 0
        self.total_detections = 0


class DistributedThreatMonitor:
    """
    Distributed threat monitoring system
    Monitors multiple nodes and aggregates threat intelligence
    """
    def __init__(self, detectors):
        """
        Args:
            detectors: Dictionary of {node_id: ThreatDetector}
        """
        self.detectors = detectors
        self.alert_history = []

    def monitor_network(self, network_data):
        """
        Monitor the entire network for threats

        Args:
            network_data: Dictionary of {node_id: data_sample}

        Returns:
            Network-wide threat report
        """
        node_results = {}
        threats_by_node = {}

        for node_id, data in network_data.items():
            if node_id in self.detectors:
                result = self.detectors[node_id].detect_threat(
                    text=data.get('text'),
                    image=data.get('image'),
                    sensor_data=data.get('sensor_data')
                )
                node_results[node_id] = result

                if result['is_threat']:
                    threats_by_node[node_id] = result

        # Generate network report
        report = {
            'timestamp': time.time(),
            'nodes_monitored': len(node_results),
            'threats_detected': len(threats_by_node),
            'node_results': node_results,
            'threat_nodes': list(threats_by_node.keys()),
            'network_threat_level': self._calculate_network_threat_level(node_results),
        }

        # Store alert if threats detected
        if threats_by_node:
            self.alert_history.append(report)

        return report

    def _calculate_network_threat_level(self, node_results):
        """
        Calculate overall network threat level
        """
        if not node_results:
            return "SAFE"

        threat_probs = [r['threat_probability'] for r in node_results.values()]
        avg_threat_prob = np.mean(threat_probs)

        if avg_threat_prob < 0.2:
            return "SAFE"
        elif avg_threat_prob < 0.4:
            return "ELEVATED"
        elif avg_threat_prob < 0.6:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_alert_history(self, limit=10):
        """Get recent alerts"""
        return self.alert_history[-limit:]
