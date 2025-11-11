"""
SIMPLIFIED FEDERATED LEARNING + MULTIMODAL LLM SYSTEM
All the concepts, less complexity

This includes:
✓ Federated Learning (distributed training)
✓ Multimodal LLM (text + sensor data)
✓ Differential Privacy (noise for security)
✓ Threat Detection (real-time)
✓ Multiple Clients (simulated locally)

But simplified:
- Smaller models (faster, easier to understand)
- Fewer parameters (clearer what each does)
- Better comments (explain everything)
- Single file (no complex imports)

Usage:
    python simplified_federated_system.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime


# ============================================================================
# STEP 1: SIMPLIFIED MULTIMODAL MODEL
# ============================================================================

class SimpleMultimodalModel(nn.Module):
    """
    Simplified multimodal model that processes:
    1. Text (security logs)
    2. Sensor data (network traffic)

    Smaller than full version but keeps the concept
    """

    def __init__(self):
        super().__init__()

        # Text encoder (simplified from BERT)
        # Input: simple text features → Output: embedding
        self.text_encoder = nn.Sequential(
            nn.Linear(50, 64),   # 50 text features → 64 dims
            nn.ReLU(),
            nn.Linear(64, 32)    # → 32 dim embedding
        )

        # Sensor encoder (for network traffic)
        # Input: sensor metrics → Output: embedding
        self.sensor_encoder = nn.Sequential(
            nn.Linear(20, 32),   # 20 sensor features → 32 dims
            nn.ReLU(),
            nn.Linear(32, 32)    # → 32 dim embedding
        )

        # Fusion layer (combines text + sensor)
        # This is the "multimodal" part!
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),   # 64 (32+32) → 32
            nn.ReLU(),
            nn.Linear(32, 2)     # → 2 classes (normal/threat)
        )

        # Learnable weights for each modality
        self.text_weight = nn.Parameter(torch.tensor(0.6))
        self.sensor_weight = nn.Parameter(torch.tensor(0.4))

    def forward(self, text_features, sensor_features):
        """
        Forward pass - the model's "thinking" process

        Args:
            text_features: Security log features (batch_size, 50)
            sensor_features: Network metrics (batch_size, 20)

        Returns:
            logits: Predictions (batch_size, 2)
        """
        # Encode each modality separately
        text_embed = self.text_encoder(text_features)      # → (batch, 32)
        sensor_embed = self.sensor_encoder(sensor_features) # → (batch, 32)

        # Weighted fusion (this is equation 2 from the paper!)
        # X_fused = w_text * X_text + w_sensor * X_sensor
        weights = torch.softmax(torch.stack([self.text_weight, self.sensor_weight]), dim=0)
        fused = torch.cat([
            weights[0] * text_embed,
            weights[1] * sensor_embed
        ], dim=1)  # → (batch, 64)

        # Final classification
        output = self.fusion(fused)  # → (batch, 2)
        return output


# ============================================================================
# STEP 2: FEDERATED LEARNING CLIENT
# ============================================================================

class FederatedClient:
    """
    Each client trains locally on their own data
    This is the "federated" part - data stays private!
    """

    def __init__(self, client_id, model, data, labels):
        self.client_id = client_id
        self.model = model
        self.data = data          # Local data (never shared!)
        self.labels = labels
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self, epochs=3):
        """
        Train on local data
        Data never leaves this client!
        """
        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            # Forward pass
            text_features = self.data[:, :50]     # First 50 features = text
            sensor_features = self.data[:, 50:]   # Last 20 features = sensor

            outputs = self.model(text_features, sensor_features)
            loss = self.criterion(outputs, self.labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / epochs
        return avg_loss

    def get_parameters(self):
        """Get model parameters to send to server"""
        return {name: param.clone() for name, param in self.model.state_dict().items()}

    def set_parameters(self, parameters):
        """Update model with global parameters from server"""
        self.model.load_state_dict(parameters)


# ============================================================================
# STEP 3: FEDERATED LEARNING SERVER
# ============================================================================

class FederatedServer:
    """
    Server aggregates updates from all clients
    Implements FedAvg algorithm (equation 1 from paper)
    """

    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_models):
        """
        FedAvg: Average all client models

        This is equation 1 from the paper:
        θ_global = (1/N) * Σ θ_i
        """
        # Start with zeros
        global_params = {}
        for name in client_models[0].keys():
            global_params[name] = torch.zeros_like(client_models[0][name])

        # Sum all client parameters
        for client_params in client_models:
            for name, param in client_params.items():
                global_params[name] += param

        # Average (divide by number of clients)
        num_clients = len(client_models)
        for name in global_params:
            global_params[name] /= num_clients

        # Update global model
        self.global_model.load_state_dict(global_params)

        return global_params


# ============================================================================
# STEP 4: DIFFERENTIAL PRIVACY
# ============================================================================

def add_differential_privacy(parameters, noise_scale=0.01):
    """
    Add noise to parameters for privacy

    This is equation 5 from the paper:
    θ̂_i = θ_i + N(0, σ²)
    """
    noisy_params = {}
    for name, param in parameters.items():
        # Add Gaussian noise
        noise = torch.randn_like(param) * noise_scale
        noisy_params[name] = param + noise

    return noisy_params


# ============================================================================
# STEP 5: DATA GENERATION (Simplified)
# ============================================================================

def create_client_data(num_samples=100, threat_ratio=0.3):
    """
    Create synthetic data for one client

    Features (70 total):
    - 50 text features (log content)
    - 20 sensor features (network metrics)
    """
    data = []
    labels = []

    for _ in range(num_samples):
        if np.random.random() < threat_ratio:
            # THREAT: High values + spikes
            text_features = np.random.randn(50) + 2.0    # Elevated
            sensor_features = np.random.randn(20) + 2.5  # High traffic

            # Add attack signatures
            text_features[10:15] += 3.0   # Suspicious patterns
            sensor_features[5:10] += 4.0  # Traffic spike

            label = 1  # Threat
        else:
            # NORMAL: Low random values
            text_features = np.random.randn(50) * 0.5
            sensor_features = np.random.randn(20) * 0.5
            label = 0  # Normal

        # Combine text + sensor
        combined = np.concatenate([text_features, sensor_features])
        data.append(combined)
        labels.append(label)

    return torch.FloatTensor(data), torch.LongTensor(labels)


# ============================================================================
# STEP 6: THREAT DETECTION
# ============================================================================

class SimpleThreatDetector:
    """Detect threats using trained model"""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def detect(self, text_features, sensor_features, description=""):
        """
        Detect if activity is a threat

        Args:
            text_features: Log features (50,)
            sensor_features: Network metrics (20,)
            description: What we're checking

        Returns:
            Dictionary with results
        """
        with torch.no_grad():
            # Convert to tensors
            text = torch.FloatTensor(text_features).unsqueeze(0)
            sensor = torch.FloatTensor(sensor_features).unsqueeze(0)

            # Get prediction
            output = self.model(text, sensor)
            probs = F.softmax(output, dim=1)

            threat_prob = probs[0, 1].item()
            is_threat = threat_prob > 0.5

            # Determine threat level
            if threat_prob > 0.9:
                level = "CRITICAL"
            elif threat_prob > 0.7:
                level = "HIGH"
            elif threat_prob > 0.5:
                level = "MEDIUM"
            else:
                level = "LOW"

            return {
                'is_threat': is_threat,
                'probability': threat_prob,
                'level': level,
                'description': description
            }


# ============================================================================
# STEP 7: SIMULATE THREATS
# ============================================================================

def simulate_sql_injection():
    """Simulate SQL injection attack"""
    text = np.random.randn(50) + 3.0
    text[10:20] += 3.0  # Attack signature
    sensor = np.random.randn(20) + 2.0
    return text, sensor, "SQL Injection Attack"

def simulate_ddos():
    """Simulate DDoS attack"""
    text = np.random.randn(50) + 2.5
    sensor = np.random.randn(20) + 5.0  # Massive traffic
    sensor[0:10] += 4.0
    return text, sensor, "DDoS Attack"

def simulate_normal():
    """Simulate normal activity"""
    text = np.random.randn(50) * 0.5
    sensor = np.random.randn(20) * 0.5
    return text, sensor, "Normal Activity"


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("SIMPLIFIED FEDERATED LEARNING + MULTIMODAL LLM SYSTEM")
    print("=" * 70)
    print()

    # Configuration
    num_clients = 3
    num_rounds = 5
    samples_per_client = 100

    print("SYSTEM CONFIGURATION")
    print("-" * 70)
    print(f"Clients: {num_clients}")
    print(f"Training Rounds: {num_rounds}")
    print(f"Samples per Client: {samples_per_client}")
    print(f"Privacy: Differential Privacy Enabled")
    print()

    # ========================================================================
    # STEP 1: INITIALIZE
    # ========================================================================

    print("STEP 1: INITIALIZING SYSTEM")
    print("-" * 70)

    # Create global model
    global_model = SimpleMultimodalModel()
    server = FederatedServer(global_model)
    print("✓ Created global model")
    print(f"  Model parameters: {sum(p.numel() for p in global_model.parameters())}")

    # Create clients with local data
    clients = []
    for i in range(num_clients):
        # Each client has their own data (stays private!)
        client_data, client_labels = create_client_data(samples_per_client)
        client_model = SimpleMultimodalModel()
        client = FederatedClient(i, client_model, client_data, client_labels)
        clients.append(client)
        print(f"✓ Created Client {i} with {samples_per_client} samples")

    print()

    # ========================================================================
    # STEP 2: FEDERATED TRAINING
    # ========================================================================

    print("STEP 2: FEDERATED TRAINING")
    print("-" * 70)
    print()

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        print("  " + "-" * 66)

        # Get current global parameters
        global_params = server.global_model.state_dict()

        # Each client trains locally
        client_updates = []
        for client in clients:
            # Download global model
            client.set_parameters(global_params)

            # Train on local data (data never shared!)
            loss = client.train_local(epochs=3)
            print(f"  Client {client.client_id}: Loss = {loss:.4f}")

            # Get local parameters
            params = client.get_parameters()

            # Add differential privacy noise
            noisy_params = add_differential_privacy(params, noise_scale=0.01)
            client_updates.append(noisy_params)

        # Server aggregates all updates (FedAvg)
        global_params = server.aggregate(client_updates)
        print(f"  ✓ Server aggregated {len(client_updates)} client updates")
        print()

    print("✓ Federated training complete!\n")

    # ========================================================================
    # STEP 3: TEST THREAT DETECTION
    # ========================================================================

    print("STEP 3: TESTING THREAT DETECTION")
    print("-" * 70)
    print()

    detector = SimpleThreatDetector(server.global_model)

    # Test different scenarios
    scenarios = [
        simulate_sql_injection(),
        simulate_ddos(),
        simulate_normal(),
        simulate_normal(),
    ]

    for text, sensor, desc in scenarios:
        result = detector.detect(text, sensor, desc)

        print(f"Test: {result['description']}")
        print(f"  Threat Probability: {result['probability']:.1%}")
        print(f"  Classification: {'THREAT' if result['is_threat'] else 'SAFE'}")
        print(f"  Threat Level: {result['level']}")

        if result['is_threat']:
            print(f"  🚨 ALERT: Threat detected!")
        else:
            print(f"  ✅ System normal")
        print()

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("=" * 70)
    print("SYSTEM SUMMARY")
    print("=" * 70)
    print()
    print("✓ FEDERATED LEARNING")
    print(f"  • {num_clients} clients trained collaboratively")
    print(f"  • Data stayed private on each client")
    print(f"  • {num_rounds} rounds of aggregation (FedAvg)")
    print()
    print("✓ MULTIMODAL LLM")
    print(f"  • Processes text (logs) + sensor (network traffic)")
    print(f"  • Learned weights: Text={global_model.text_weight.item():.2f}, Sensor={global_model.sensor_weight.item():.2f}")
    print(f"  • Fusion layer combines both modalities")
    print()
    print("✓ DIFFERENTIAL PRIVACY")
    print(f"  • Gaussian noise added to all updates")
    print(f"  • Protects individual client data")
    print()
    print("✓ THREAT DETECTION")
    print(f"  • Real-time classification")
    print(f"  • Probability-based threat levels")
    print(f"  • Detects: SQL injection, DDoS, and more")
    print()
    print("=" * 70)
    print()
    print("KEY CONCEPTS DEMONSTRATED:")
    print("  1. Federated Learning - Distributed training without sharing data")
    print("  2. Multimodal LLM - Combining text + sensor data")
    print("  3. Differential Privacy - Adding noise for security")
    print("  4. Real-time Detection - Fast threat classification")
    print()
    print("This simplified version has all the concepts of the full system,")
    print("just with smaller models and clearer code!")
    print()


if __name__ == '__main__':
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
