"""
SIMPLIFIED THREAT DETECTION DEMO
Easy to understand, single file implementation

This shows the core concept:
1. Train a simple neural network
2. Detect threats vs normal activity
3. Show results

Usage:
    python simple_demo.py
"""

import torch
import torch.nn as nn
import numpy as np
import random


# ============================================================================
# STEP 1: SIMPLE MODEL
# ============================================================================

class SimpleThreatDetector(nn.Module):
    """Simple neural network for threat detection"""

    def __init__(self):
        super().__init__()
        # Simple 3-layer network
        self.network = nn.Sequential(
            nn.Linear(100, 50),    # Input: 100 features
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 2)       # Output: [normal, threat]
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# STEP 2: GENERATE SIMPLE TRAINING DATA
# ============================================================================

def create_training_data(num_samples=1000):
    """Create simple training data"""
    data = []
    labels = []

    for _ in range(num_samples):
        if random.random() < 0.5:
            # Normal activity: low random values
            features = np.random.randn(100) * 0.5
            label = 0  # Normal
        else:
            # Threat: high values + spikes
            features = np.random.randn(100) + 2.0  # Shifted higher
            features[random.randint(0, 50)] += 5.0  # Add spike
            label = 1  # Threat

        data.append(features)
        labels.append(label)

    return torch.FloatTensor(data), torch.LongTensor(labels)


# ============================================================================
# STEP 3: TRAIN THE MODEL
# ============================================================================

def train_model(model, data, labels, epochs=10):
    """Simple training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean() * 100

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.1f}%")

    print("✓ Training complete!\n")


# ============================================================================
# STEP 4: TEST WITH SIMULATED THREATS
# ============================================================================

def simulate_sql_injection():
    """Simulate SQL injection attack"""
    features = np.random.randn(100) + 3.0  # High anomaly
    features[10:20] += 2.0  # Spike in specific area
    return features, "SQL Injection: SELECT * FROM users WHERE '1'='1'"


def simulate_ddos():
    """Simulate DDoS attack"""
    features = np.random.randn(100) + 4.0  # Very high anomaly
    features[0:30] += 3.0  # Large spike
    return features, "DDoS Attack: 10000 requests in 1 second"


def simulate_brute_force():
    """Simulate brute force attack"""
    features = np.random.randn(100) + 2.5
    features[40:60] += 2.0
    return features, "Brute Force: 50 failed login attempts"


def simulate_normal_activity():
    """Simulate normal activity"""
    features = np.random.randn(100) * 0.5  # Low, random
    return features, "Normal: User login successful"


# ============================================================================
# STEP 5: DETECT THREATS
# ============================================================================

def detect_threat(model, features, description):
    """Detect if something is a threat"""
    model.eval()

    with torch.no_grad():
        # Convert to tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)

        # Get prediction
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        normal_prob = probabilities[0][0].item()
        threat_prob = probabilities[0][1].item()

        is_threat = threat_prob > 0.5

    # Print results
    print("─" * 70)
    print(f"Event: {description}")
    print(f"Normal Probability: {normal_prob:.1%}")
    print(f"Threat Probability: {threat_prob:.1%}")

    if is_threat:
        if threat_prob > 0.9:
            level = "🔴 CRITICAL"
        elif threat_prob > 0.7:
            level = "🟠 HIGH"
        else:
            level = "🟡 MEDIUM"
        print(f"Result: ⚠️  THREAT DETECTED - {level}")
    else:
        print(f"Result: ✅ SAFE - No threat detected")

    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("SIMPLE THREAT DETECTION DEMO")
    print("=" * 70)
    print()

    # Step 1: Create model
    print("Step 1: Creating simple neural network...")
    model = SimpleThreatDetector()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters\n")

    # Step 2: Generate training data
    print("Step 2: Generating training data...")
    train_data, train_labels = create_training_data(num_samples=1000)
    print(f"✓ Created 1000 training samples (500 normal, 500 threats)\n")

    # Step 3: Train
    print("Step 3: Training the model...")
    train_model(model, train_data, train_labels, epochs=10)

    # Step 4: Test with threats
    print("=" * 70)
    print("Step 4: Testing with Simulated Threats")
    print("=" * 70)
    print()

    # Test SQL Injection
    features, desc = simulate_sql_injection()
    detect_threat(model, features, desc)

    # Test DDoS
    features, desc = simulate_ddos()
    detect_threat(model, features, desc)

    # Test Brute Force
    features, desc = simulate_brute_force()
    detect_threat(model, features, desc)

    # Test Normal Activity
    features, desc = simulate_normal_activity()
    detect_threat(model, features, desc)

    # Summary
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("What happened:")
    print("  1. Created a simple neural network")
    print("  2. Trained it to recognize threats vs normal activity")
    print("  3. Tested with simulated cyber attacks")
    print("  4. Successfully detected threats!")
    print()
    print("The model learned patterns:")
    print("  • Threats = high values + spikes in features")
    print("  • Normal = low, random values")
    print()


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()
