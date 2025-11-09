"""
Federated Learning Client for Remote Machines
Run this on each client machine

Usage:
    python client_node.py --client-id client_1 --server-url http://192.168.1.100:8000
"""

import torch
import requests
import pickle
import base64
import time
import argparse
from datetime import datetime
from models.multimodal_model import MultimodalFusionModel
from federated.client import FederatedClient
from data.dataset import create_synthetic_data, MultimodalSecurityDataset
from config import MODEL_CONFIG, FL_CONFIG


class RemoteFederatedClient:
    """Client that communicates with remote server via REST API"""

    def __init__(self, client_id, server_url, train_dataset, device='cpu'):
        self.client_id = client_id
        self.server_url = server_url
        self.train_dataset = train_dataset

        # Initialize local model
        self.model = MultimodalFusionModel(MODEL_CONFIG)
        self.client = FederatedClient(
            client_id=client_id,
            model=self.model,
            train_data=train_dataset,
            device=device
        )

        print(f"\n{'='*70}")
        print(f"Client {client_id} Initialized")
        print(f"{'='*70}")
        print(f"Server URL: {server_url}")
        print(f"Device: {self.client.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Local epochs: {FL_CONFIG['local_epochs']}")
        print(f"Batch size: {FL_CONFIG['batch_size']}")
        print(f"{'='*70}\n")

    def check_server(self):
        """Check if server is accessible"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Server is online (Round {data['round']}/{data['max_rounds']})")
            return True
        except Exception as e:
            print(f"✗ Cannot reach server: {e}")
            return False

    def get_global_model(self):
        """Download global model from server"""
        try:
            response = requests.get(
                f"{self.server_url}/get_global_model",
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Deserialize parameters
            encoded = data['model_parameters']
            serialized = base64.b64decode(encoded)
            parameters = pickle.loads(serialized)

            # Update local model
            self.client.set_parameters(parameters)

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ✓ Downloaded global model (Round {data['round']})")
            return data['round']

        except requests.exceptions.Timeout:
            print(f"✗ Timeout downloading model")
            return None
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return None

    def train_local(self):
        """Train on local data"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] → Training on local data...")

        start_time = time.time()
        loss = self.client.train_local(epochs=FL_CONFIG['local_epochs'])
        training_time = time.time() - start_time

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ✓ Training complete | Loss: {loss:.4f} | Time: {training_time:.1f}s")

        return loss

    def upload_update(self, loss):
        """Upload model update to server"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] → Uploading model update...")

            # Get local parameters
            parameters = self.client.get_parameters()

            # Serialize and encode
            serialized = pickle.dumps(parameters)
            encoded = base64.b64encode(serialized).decode('utf-8')

            # Upload
            payload = {
                'client_id': self.client_id,
                'model_parameters': encoded,
                'num_samples': len(self.train_dataset),
                'loss': loss
            }

            response = requests.post(
                f"{self.server_url}/upload_update",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ✓ Upload successful (Pending: {data.get('pending_clients', 0)})")

            return True

        except requests.exceptions.Timeout:
            print(f"✗ Timeout uploading update")
            return False
        except Exception as e:
            print(f"✗ Error uploading update: {e}")
            return False

    def wait_for_aggregation(self, wait_time=30):
        """Wait for server to aggregate"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Waiting {wait_time}s for aggregation...")
        time.sleep(wait_time)

    def run_training_loop(self, num_rounds=10, wait_time=30):
        """Run federated learning training loop"""
        print("\n" + "="*70)
        print(f"Starting Federated Training")
        print(f"Client: {self.client_id}")
        print(f"Rounds: {num_rounds}")
        print("="*70 + "\n")

        # Check server connectivity
        if not self.check_server():
            print("\n❌ Cannot connect to server. Exiting.")
            return

        successful_rounds = 0

        for round_num in range(num_rounds):
            print(f"\n{'─'*70}")
            print(f"ROUND {round_num + 1}/{num_rounds}")
            print(f"{'─'*70}")

            # Step 1: Download global model
            current_round = self.get_global_model()
            if current_round is None:
                print("⚠️ Failed to get global model, waiting 10s before retry...")
                time.sleep(10)
                continue

            # Step 2: Train locally
            try:
                loss = self.train_local()
            except Exception as e:
                print(f"✗ Training failed: {e}")
                continue

            # Step 3: Upload update
            success = self.upload_update(loss)
            if not success:
                print("⚠️ Failed to upload update, waiting 10s before retry...")
                time.sleep(10)
                continue

            successful_rounds += 1

            # Step 4: Wait for aggregation
            self.wait_for_aggregation(wait_time)

        print("\n" + "="*70)
        print(f"Training Complete!")
        print(f"Client: {self.client_id}")
        print(f"Successful Rounds: {successful_rounds}/{num_rounds}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', required=True,
                       help='Unique client ID (e.g., client_1, hospital_a)')
    parser.add_argument('--server-url', required=True,
                       help='Server URL (e.g., http://192.168.1.100:8000)')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of training rounds')
    parser.add_argument('--data-size', type=int, default=1000,
                       help='Number of local training samples')
    parser.add_argument('--wait-time', type=int, default=30,
                       help='Wait time (seconds) between rounds')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create local dataset
    print("\nGenerating local training data...")
    local_data = create_synthetic_data(
        num_samples=args.data_size,
        threat_ratio=0.3
    )
    train_dataset = MultimodalSecurityDataset(local_data)
    print(f"✓ Created {len(train_dataset)} training samples")

    # Create and run client
    client = RemoteFederatedClient(
        client_id=args.client_id,
        server_url=args.server_url,
        train_dataset=train_dataset,
        device=device
    )

    try:
        client.run_training_loop(
            num_rounds=args.num_rounds,
            wait_time=args.wait_time
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
