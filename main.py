"""
Main script to run the Distributed Security Threat Detection System
Integrating Federated Learning and Multimodal LLM
"""

import torch
import numpy as np
import time
from torch.utils.data import DataLoader

# Import project modules
from models.multimodal_model import MultimodalFusionModel
from federated.client import FederatedClient
from federated.server import FederatedServer
from data.dataset import ThreatDetectionDataModule
from detection.threat_detector import ThreatDetector, DistributedThreatMonitor
from config import FL_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG

# Set random seeds for reproducibility
torch.manual_seed(SYSTEM_CONFIG['random_seed'])
np.random.seed(SYSTEM_CONFIG['random_seed'])


def main():
    print("=" * 70)
    print("Distributed Security Threat Detection System")
    print("Federated Learning + Multimodal LLM")
    print("=" * 70)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() and SYSTEM_CONFIG['use_gpu'] else 'cpu'
    print(f"\nUsing device: {device}")

    # Step 1: Create data module
    print("\n[Step 1/5] Preparing multimodal security dataset...")
    data_module = ThreatDetectionDataModule(
        num_samples=5000,  # Reduced for demo
        num_clients=FL_CONFIG['num_clients'],
        threat_ratio=0.3
    )

    # Step 2: Initialize global model
    print("\n[Step 2/5] Initializing multimodal fusion model...")
    global_model = MultimodalFusionModel(MODEL_CONFIG)
    print(f"Model initialized with {sum(p.numel() for p in global_model.parameters()):,} parameters")

    # Step 3: Create federated clients
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
    print(f"Created {len(clients)} clients successfully")

    # Step 4: Initialize federated server
    print("\n[Step 4/5] Initializing federated learning server...")
    server = FederatedServer(global_model, device=device)

    # Step 5: Run federated learning
    print(f"\n[Step 5/5] Starting federated learning training...")
    print(f"Training for {FL_CONFIG['num_rounds']} rounds with {FL_CONFIG['local_epochs']} local epochs per round")

    start_time = time.time()

    # Train for fewer rounds in demo
    num_rounds = min(10, FL_CONFIG['num_rounds'])
    training_history = server.train(clients, num_rounds=num_rounds)

    training_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("Federated Learning Training Complete!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per round: {training_time/num_rounds:.2f} seconds")
    print(f"{'='*70}")

    # Evaluate the global model
    print("\n[Evaluation] Evaluating global model on test set...")
    test_dataset = data_module.get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
        print(f"  Status: {'✓ CORRECT' if (result['is_threat'] and test_case['expected'] == 'threat') or (not result['is_threat'] and test_case['expected'] == 'normal') else '✗ INCORRECT'}")

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

    # Demonstrate distributed monitoring
    print("\n[Demo] Distributed Network Monitoring...")

    # Create detectors for multiple nodes
    node_detectors = {
        f"node_{i}": ThreatDetector(global_model, device=device)
        for i in range(5)
    }

    monitor = DistributedThreatMonitor(node_detectors)

    # Simulate network data
    network_data = {
        "node_0": {'text': "Normal system operation", 'sensor': np.random.randn(100)},
        "node_1": {'text': "Unusual port scanning detected", 'sensor': np.random.randn(100) + 2.0},
        "node_2": {'text': "Database connection established", 'sensor': np.random.randn(100)},
        "node_3": {'text': "Potential DDoS attack detected", 'sensor': np.random.randn(100) + 3.0},
        "node_4": {'text': "Backup completed successfully", 'sensor': np.random.randn(100)},
    }

    report = monitor.monitor_network(network_data)

    print(f"\n{'='*70}")
    print("DISTRIBUTED NETWORK MONITORING REPORT")
    print(f"{'='*70}")
    print(f"Nodes Monitored:       {report['nodes_monitored']}")
    print(f"Threats Detected:      {report['threats_detected']}")
    print(f"Network Threat Level:  {report['network_threat_level']}")
    print(f"Threat Nodes:          {', '.join(report['threat_nodes']) if report['threat_nodes'] else 'None'}")
    print(f"{'='*70}")

    print("\n" + "="*70)
    print("SYSTEM SUMMARY")
    print("="*70)
    print(f"✓ Federated learning with {FL_CONFIG['num_clients']} distributed nodes")
    print(f"✓ Multimodal LLM processing text, images, and sensor data")
    print(f"✓ Differential privacy protection enabled")
    print(f"✓ Real-time threat detection: ~{stats['avg_detection_time_ms']:.2f} ms/sample")
    print(f"✓ Detection accuracy: {metrics['accuracy']:.2f}%")
    print(f"✓ Privacy-preserving distributed architecture")
    print("="*70)

    print("\n✓ Prototype demonstration complete!")
    print("\nNext steps:")
    print("  1. Integrate real security datasets (logs, images, network traffic)")
    print("  2. Deploy to production distributed environment")
    print("  3. Implement continuous learning pipeline")
    print("  4. Add alerting and incident response system")
    print("  5. Scale to larger number of nodes")

    return {
        'server': server,
        'clients': clients,
        'detector': detector,
        'metrics': metrics,
        'training_history': training_history
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
