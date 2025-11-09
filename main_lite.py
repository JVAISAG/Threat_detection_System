"""
Memory-optimized version for systems with limited GPU memory
Uses CPU and smaller configurations
"""

import torch
import numpy as np
import time
import gc
from torch.utils.data import DataLoader

# Import project modules
from models.multimodal_model import MultimodalFusionModel
from federated.client import FederatedClient
from federated.server import FederatedServer
from data.dataset import ThreatDetectionDataModule
from detection.threat_detector import ThreatDetector
from config import FL_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG

# Set random seeds for reproducibility
torch.manual_seed(SYSTEM_CONFIG['random_seed'])
np.random.seed(SYSTEM_CONFIG['random_seed'])


def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("Distributed Security Threat Detection System (Lite Version)")
    print("Optimized for Limited Memory")
    print("=" * 70)

    # Force CPU usage to avoid GPU memory issues
    device = 'cpu'
    print(f"\nUsing device: {device}")
    print("Note: Running on CPU to avoid GPU memory issues")

    # Step 1: Create smaller dataset
    print("\n[Step 1/5] Preparing multimodal security dataset...")
    data_module = ThreatDetectionDataModule(
        num_samples=1000,  # Reduced from 5000
        num_clients=FL_CONFIG['num_clients'],
        threat_ratio=0.3
    )

    # Step 2: Initialize global model
    print("\n[Step 2/5] Initializing multimodal fusion model...")
    global_model = MultimodalFusionModel(MODEL_CONFIG)
    print(f"Model initialized with {sum(p.numel() for p in global_model.parameters()):,} parameters")

    # Step 3: Create federated clients (one at a time to save memory)
    print(f"\n[Step 3/5] Creating {FL_CONFIG['num_clients']} federated learning clients...")
    clients = []
    for i in range(FL_CONFIG['num_clients']):
        client_dataset = data_module.get_client_dataset(i)
        client_model = MultimodalFusionModel(MODEL_CONFIG)
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_data=client_dataset,
            device=device
        )
        clients.append(client)
        print(f"  Created client {i+1}/{FL_CONFIG['num_clients']}")

    clear_memory()

    # Step 4: Initialize federated server
    print("\n[Step 4/5] Initializing federated learning server...")
    server = FederatedServer(global_model, device=device)

    # Step 5: Run federated learning with memory management
    print(f"\n[Step 5/5] Starting federated learning training...")
    num_rounds = 5  # Reduced for demo
    print(f"Training for {num_rounds} rounds (reduced for memory efficiency)")

    start_time = time.time()

    # Custom training loop with memory management
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Round {round_num}/{num_rounds} ===")

        # Select subset of clients for this round
        num_clients = max(1, int(len(clients) * FL_CONFIG['client_fraction']))
        selected_clients = np.random.choice(clients, num_clients, replace=False)

        # Distribute global model
        global_params = server.get_global_parameters()
        for client in selected_clients:
            client.set_parameters(global_params)

        # Train clients sequentially to save memory
        client_parameters = []
        client_weights = []

        for client in selected_clients:
            print(f"Training Client {client.client_id}...")
            loss = client.train_local()
            params = client.get_parameters()

            client_parameters.append(params)
            client_weights.append(len(client.train_data))

            print(f"  Client {client.client_id} - Loss: {loss:.4f}")

            # Clear memory after each client
            clear_memory()

        # Aggregate updates
        server.update_global_model(client_parameters, client_weights)
        clear_memory()

    training_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("Federated Learning Training Complete!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per round: {training_time/num_rounds:.2f} seconds")
    print(f"{'='*70}")

    # Evaluate the global model
    print("\n[Evaluation] Evaluating global model on test set...")
    test_dataset = data_module.get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # Small batch size

    start_eval = time.time()
    metrics = server.evaluate_global_model(test_loader, device=device)
    eval_time = time.time() - start_eval

    print(f"\n{'='*70}")
    print("GLOBAL MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy:              {metrics['accuracy']:.2f}%")
    print(f"False Positive Rate:   {metrics['false_positive_rate']:.2f}%")
    print(f"False Negative Rate:   {metrics['false_negative_rate']:.2f}%")
    print(f"Total Samples:         {metrics['total_samples']}")
    print(f"Evaluation Time:       {eval_time:.2f} seconds")
    print(f"Avg Processing Time:   {eval_time/metrics['total_samples']*1000:.2f} ms/sample")
    print(f"{'='*70}")

    # Demonstrate real-time threat detection
    print("\n[Demo] Real-time Threat Detection Demo...")
    detector = ThreatDetector(global_model, device=device, threshold=0.5)

    # Test cases
    test_cases = [
        {
            'text': "User login successful from IP 192.168.1.100",
            'sensor': np.random.randn(100),
            'expected': 'normal'
        },
        {
            'text': "Multiple failed login attempts detected from IP 10.0.0.50",
            'sensor': np.random.randn(100) + 2.0,
            'expected': 'threat'
        },
        {
            'text': "SQL injection attempt in web request",
            'sensor': np.random.randn(100) + 2.5,
            'expected': 'threat'
        },
    ]

    print(f"\n{'='*70}")
    print("REAL-TIME THREAT DETECTION RESULTS")
    print(f"{'='*70}")

    for i, test_case in enumerate(test_cases):
        result = detector.detect_threat(
            text=test_case['text'],
            sensor_data=test_case['sensor']
        )

        print(f"\nTest Case {i+1}:")
        print(f"  Input: {test_case['text'][:60]}...")
        print(f"  Expected: {test_case['expected'].upper()}")
        print(f"  Detection: {'THREAT' if result['is_threat'] else 'NORMAL'}")
        print(f"  Threat Probability: {result['threat_probability']:.2%}")
        print(f"  Threat Level: {result['threat_level']}")
        print(f"  Detection Time: {result['detection_time_ms']:.2f} ms")

    # Display detection statistics
    stats = detector.get_statistics()
    print(f"\n{'='*70}")
    print("DETECTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total Detections:      {stats['total_detections']}")
    print(f"Threats Detected:      {stats['threats_detected']}")
    print(f"Threat Rate:           {stats['threat_rate_percent']:.2f}%")
    print(f"Avg Detection Time:    {stats['avg_detection_time_ms']:.2f} ms")
    print(f"{'='*70}")

    print("\n" + "="*70)
    print("SYSTEM SUMMARY")
    print("="*70)
    print(f"✓ Federated learning with {FL_CONFIG['num_clients']} distributed nodes")
    print(f"✓ Memory-optimized configuration")
    print(f"✓ Running on CPU to avoid GPU memory issues")
    print(f"✓ Real-time threat detection: ~{stats['avg_detection_time_ms']:.2f} ms/sample")
    print(f"✓ Detection accuracy: {metrics['accuracy']:.2f}%")
    print(f"✓ Privacy-preserving distributed architecture")
    print("="*70)

    print("\n✓ Lite prototype demonstration complete!")
    print("\nMemory optimization tips:")
    print("  1. System now uses CPU instead of GPU")
    print("  2. Reduced batch size to 4 (was 32)")
    print("  3. Reduced number of clients to 5 (was 10)")
    print("  4. Training only 40% of clients per round")
    print("  5. Memory cleared after each training step")
    print("\nTo use GPU with more memory:")
    print("  - Set use_gpu=True in config.py")
    print("  - Increase GPU memory limit")
    print("  - Or use a cloud GPU instance with more memory")

    return {
        'server': server,
        'clients': clients,
        'detector': detector,
        'metrics': metrics,
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        clear_memory()
