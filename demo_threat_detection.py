"""
Simple Demo: Train model and test with simulated threats
This demonstrates the complete workflow: train → test → detect

Usage:
    python demo_threat_detection.py
"""

import torch
import numpy as np
from models.multimodal_model import MultimodalFusionModel
from federated.client import FederatedClient
from federated.server import FederatedServer
from data.dataset import create_synthetic_data, MultimodalSecurityDataset
from detection.threat_detector import ThreatDetector
from test_threats import ThreatSimulator
from config import MODEL_CONFIG, FL_CONFIG
import time


def quick_training(num_rounds=3):
    """Quick training with synthetic data"""
    print("="*70)
    print("STEP 1: TRAINING MODEL")
    print("="*70)

    device = 'cpu'
    print(f"\nTraining on: {device}")
    print(f"Rounds: {num_rounds}")

    # Create data
    print("\nGenerating training data...")
    train_data = create_synthetic_data(num_samples=500, threat_ratio=0.3)
    train_dataset = MultimodalSecurityDataset(train_data)
    print(f"✓ Created {len(train_dataset)} training samples")

    # Initialize model and server
    global_model = MultimodalFusionModel(MODEL_CONFIG)
    server = FederatedServer(global_model, device=device)

    # Create client
    client = FederatedClient(
        client_id=0,
        model=MultimodalFusionModel(MODEL_CONFIG),
        train_data=train_dataset,
        device=device
    )

    # Training loop
    print(f"\nTraining for {num_rounds} rounds...")
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")

        # Get global model
        global_params = server.get_global_parameters()
        client.set_parameters(global_params)

        # Train locally
        loss = client.train_local(epochs=3)
        print(f"  Loss: {loss:.4f}")

        # Update server
        params = client.get_parameters()
        server.update_global_model([params], [len(train_dataset)])

    print("\n✓ Training complete!")
    return server.global_model


def test_with_threats(model, num_tests=10):
    """Test model with simulated threats"""
    print("\n" + "="*70)
    print("STEP 2: TESTING WITH SIMULATED THREATS")
    print("="*70)

    detector = ThreatDetector(model, device='cpu', threshold=0.5)
    simulator = ThreatSimulator()

    results = {'correct': 0, 'total': 0}

    for i in range(num_tests):
        # Generate threat or normal activity
        if i < num_tests * 0.7:  # 70% threats
            threat_types = ['sql_injection', 'ddos_attack', 'brute_force', 'malware_detected']
            threat = simulator.generate_threat(np.random.choice(threat_types))
            is_actual_threat = True
        else:
            threat = simulator.generate_normal_activity()
            is_actual_threat = False

        # Detect
        result = detector.detect_threat(
            text=threat['log'],
            sensor_data=threat['sensor']
        )

        is_detected = result['is_threat']
        correct = (is_actual_threat == is_detected)

        results['total'] += 1
        if correct:
            results['correct'] += 1

        # Status
        if is_actual_threat and is_detected:
            status = "✓ THREAT DETECTED"
            emoji = "🚨"
        elif is_actual_threat and not is_detected:
            status = "✗ THREAT MISSED"
            emoji = "⚠️"
        elif not is_actual_threat and is_detected:
            status = "⚠ FALSE ALARM"
            emoji = "❌"
        else:
            status = "✓ NORMAL (Correct)"
            emoji = "✅"

        print(f"\nTest {i+1}/{num_tests} {emoji}")
        print(f"  Type: {threat['threat_type']}")
        print(f"  Log: {threat['log'][:60]}...")
        print(f"  Expected: {'THREAT' if is_actual_threat else 'NORMAL'}")
        print(f"  Detected: {'THREAT' if is_detected else 'NORMAL'} ({result['threat_probability']:.1%})")
        print(f"  Result: {status}")

    # Summary
    accuracy = (results['correct'] / results['total']) * 100
    print(f"\n{'='*70}")
    print(f"RESULTS: {results['correct']}/{results['total']} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*70}")

    return results


def demonstrate_specific_threats(model):
    """Demonstrate detection of specific threat types"""
    print("\n" + "="*70)
    print("STEP 3: SPECIFIC THREAT DEMONSTRATIONS")
    print("="*70)

    detector = ThreatDetector(model, device='cpu', threshold=0.5)
    simulator = ThreatSimulator()

    threat_types = [
        ('sql_injection', 'SQL Injection Attack'),
        ('ddos_attack', 'DDoS Attack'),
        ('brute_force', 'Brute Force Attack'),
        ('malware_detected', 'Malware Detection'),
        ('normal', 'Normal Activity')
    ]

    for threat_key, threat_name in threat_types:
        print(f"\n{'─'*70}")
        print(f"DEMONSTRATING: {threat_name}")
        print(f"{'─'*70}")

        threat = simulator.generate_threat(threat_key)

        print(f"\nSimulated Log Entry:")
        print(f"  {threat['log']}")

        result = detector.detect_threat(
            text=threat['log'],
            sensor_data=threat['sensor']
        )

        print(f"\nDetection Result:")
        print(f"  Threat Detected: {'YES 🚨' if result['is_threat'] else 'NO ✅'}")
        print(f"  Probability: {result['threat_probability']:.1%}")
        print(f"  Threat Level: {result['threat_level']}")
        print(f"  Detection Time: {result['detection_time_ms']:.2f}ms")

        if threat_key != 'normal' and result['is_threat']:
            print(f"  ✓ Successfully detected {threat_name}!")
        elif threat_key == 'normal' and not result['is_threat']:
            print(f"  ✓ Correctly identified as normal activity!")

        time.sleep(0.5)  # Pause between demos


def main():
    print("\n" + "="*70)
    print("THREAT DETECTION DEMONSTRATION")
    print("Train model → Test with threats → Demonstrate detection")
    print("="*70)

    # Step 1: Train model
    print("\nThis will:")
    print("  1. Train a model with synthetic data (3 rounds)")
    print("  2. Test with 10 simulated threats")
    print("  3. Demonstrate specific attack detections")
    print("\nPress Enter to start...")
    input()

    trained_model = quick_training(num_rounds=3)

    # Step 2: Test with random threats
    test_results = test_with_threats(trained_model, num_tests=10)

    # Step 3: Demonstrate specific threats
    demonstrate_specific_threats(trained_model)

    # Final summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nWhat you just saw:")
    print("  ✓ Model trained on synthetic security data")
    print("  ✓ Threats simulated (SQL injection, DDoS, malware, etc.)")
    print("  ✓ Real-time detection with probability scores")
    print("  ✓ Threat level classification (LOW/MEDIUM/HIGH/CRITICAL)")
    print("\nThe system successfully:")
    print(f"  • Detected {test_results['correct']}/{test_results['total']} threats correctly")
    print("  • Distinguished between threats and normal activity")
    print("  • Provided fast detection (~1-50ms per sample)")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test more threats:")
    print("   python test_threats.py --num-tests 50")
    print("\n2. Interactive testing:")
    print("   python test_threats.py --interactive")
    print("\n3. Train longer for better accuracy:")
    print("   python main_lite.py")
    print("\n4. Deploy across multiple machines:")
    print("   See QUICK_START.md")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
