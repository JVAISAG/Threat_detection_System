# Multi-Machine Deployment Guide
## Distributed Security Threat Detection System

This guide explains how to deploy the federated learning system across multiple physical or virtual machines.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Network Setup](#network-setup)
3. [Method 1: REST API (Recommended)](#method-1-rest-api-recommended)
4. [Method 2: gRPC](#method-2-grpc)
5. [Method 3: Docker + Kubernetes](#method-3-docker--kubernetes)
6. [Security Considerations](#security-considerations)
7. [Production Deployment](#production-deployment)

---

## Architecture Overview

### Distributed Setup
```
┌─────────────────────────────────────────────────────┐
│              Central Server (Aggregator)            │
│              IP: 192.168.1.100:8000                 │
│         - Receives model updates                    │
│         - Performs FedAvg aggregation               │
│         - Distributes global model                  │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼──────────┐ ┌▼────────────┐
│  Client 1    │ │  Client 2   │ │  Client N   │
│  Hospital    │ │  Bank       │ │  Government │
│  10.0.1.50   │ │  10.0.2.30  │ │  10.0.3.40  │
│              │ │             │ │             │
│ [Local Data] │ │[Local Data] │ │[Local Data] │
└──────────────┘ └─────────────┘ └─────────────┘
```

### Communication Flow
1. **Initialization**: Server creates global model
2. **Distribution**: Server sends model to clients
3. **Local Training**: Clients train on local data
4. **Upload**: Clients send updates to server
5. **Aggregation**: Server aggregates updates
6. **Repeat**: Loop until convergence

---

## Network Setup

### Prerequisites

**Server Requirements:**
- Python 3.8+
- 16GB+ RAM
- Network accessible to all clients
- Open ports: 8000 (REST) or 50051 (gRPC)

**Client Requirements:**
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended
- Network access to server

### Network Configuration

**Option A: Same LAN (Easiest)**
```
All machines on same network: 192.168.1.0/24
Server: 192.168.1.100
Client 1: 192.168.1.101
Client 2: 192.168.1.102
```

**Option B: VPN (Secure)**
```
Use VPN (Tailscale, WireGuard, OpenVPN)
Server: 100.64.0.1
Clients: 100.64.0.2, 100.64.0.3, ...
```

**Option C: Cloud (Scalable)**
```
Server: AWS EC2 us-east-1
Clients: Multiple regions/cloud providers
Use load balancer + security groups
```

---

## Method 1: REST API (Recommended)

This is the easiest method for multi-machine deployment.

### Step 1: Create Server API

Create `server_api.py`:

```python
"""
Federated Learning Server with REST API
Run this on the central server machine
"""

from flask import Flask, request, jsonify
import torch
import pickle
import base64
from models.multimodal_model import MultimodalFusionModel
from federated.server import FederatedServer
from config import MODEL_CONFIG
import threading
import time

app = Flask(__name__)

# Global server instance
global_model = MultimodalFusionModel(MODEL_CONFIG)
server = FederatedServer(global_model, device='cpu')

# Store client updates
client_updates = {}
update_lock = threading.Lock()

# Training state
current_round = 0
max_rounds = 50
clients_per_round = 2


@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is alive"""
    return jsonify({
        'status': 'online',
        'round': current_round,
        'clients_registered': len(client_updates)
    })


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """
    Clients request the current global model
    Returns: Serialized model parameters
    """
    global_params = server.get_global_parameters()

    # Serialize parameters
    serialized = pickle.dumps(global_params)
    encoded = base64.b64encode(serialized).decode('utf-8')

    return jsonify({
        'round': current_round,
        'model_parameters': encoded,
        'timestamp': time.time()
    })


@app.route('/upload_update', methods=['POST'])
def upload_update():
    """
    Clients upload their model updates
    Request body: {
        'client_id': 'client_1',
        'model_parameters': '<base64_encoded>',
        'num_samples': 1000,
        'loss': 0.05
    }
    """
    data = request.json
    client_id = data['client_id']

    # Deserialize parameters
    encoded_params = data['model_parameters']
    serialized = base64.b64decode(encoded_params)
    parameters = pickle.loads(serialized)

    # Store update
    with update_lock:
        client_updates[client_id] = {
            'parameters': parameters,
            'num_samples': data['num_samples'],
            'loss': data['loss'],
            'timestamp': time.time()
        }

    print(f"Received update from {client_id} (Loss: {data['loss']:.4f})")

    return jsonify({
        'status': 'success',
        'message': f'Update received from {client_id}',
        'round': current_round
    })


@app.route('/trigger_aggregation', methods=['POST'])
def trigger_aggregation():
    """
    Manually trigger aggregation when enough clients have uploaded
    In production, this could be automatic
    """
    global current_round

    with update_lock:
        if len(client_updates) < clients_per_round:
            return jsonify({
                'status': 'waiting',
                'message': f'Only {len(client_updates)}/{clients_per_round} clients ready'
            }), 400

        # Extract parameters and weights
        client_params = [u['parameters'] for u in client_updates.values()]
        client_weights = [u['num_samples'] for u in client_updates.values()]

        # Aggregate
        server.update_global_model(client_params, client_weights)
        current_round += 1

        avg_loss = sum(u['loss'] for u in client_updates.values()) / len(client_updates)

        # Clear updates for next round
        result = {
            'status': 'success',
            'round': current_round,
            'clients_aggregated': len(client_updates),
            'avg_loss': avg_loss
        }

        client_updates.clear()

        print(f"Round {current_round} complete - Avg Loss: {avg_loss:.4f}")

        return jsonify(result)


@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current training status"""
    return jsonify({
        'current_round': current_round,
        'max_rounds': max_rounds,
        'pending_clients': len(client_updates),
        'clients_per_round': clients_per_round
    })


def run_server(host='0.0.0.0', port=8000):
    """Start the federated learning server"""
    print("="*70)
    print("Federated Learning Server Starting...")
    print(f"Listening on {host}:{port}")
    print("="*70)
    app.run(host=host, port=port, threaded=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', default=8000, type=int, help='Server port')
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
```

### Step 2: Create Client Script

Create `client_node.py`:

```python
"""
Federated Learning Client for Remote Machines
Run this on each client machine
"""

import torch
import requests
import pickle
import base64
import time
import argparse
from models.multimodal_model import MultimodalFusionModel
from federated.client import FederatedClient
from data.dataset import ThreatDetectionDataModule
from config import MODEL_CONFIG, FL_CONFIG


class RemoteFederatedClient:
    """Client that communicates with remote server via REST API"""

    def __init__(self, client_id, server_url, train_dataset):
        self.client_id = client_id
        self.server_url = server_url
        self.train_dataset = train_dataset

        # Initialize local model
        self.model = MultimodalFusionModel(MODEL_CONFIG)
        self.client = FederatedClient(
            client_id=client_id,
            model=self.model,
            train_data=train_dataset,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"Client {client_id} initialized")
        print(f"Server: {server_url}")
        print(f"Device: {self.client.device}")
        print(f"Training samples: {len(train_dataset)}")

    def get_global_model(self):
        """Download global model from server"""
        try:
            response = requests.get(f"{self.server_url}/get_global_model", timeout=30)
            response.raise_for_status()

            data = response.json()

            # Deserialize parameters
            encoded = data['model_parameters']
            serialized = base64.b64decode(encoded)
            parameters = pickle.loads(serialized)

            # Update local model
            self.client.set_parameters(parameters)

            print(f"✓ Downloaded global model (Round {data['round']})")
            return data['round']

        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return None

    def train_local(self):
        """Train on local data"""
        print(f"Training client {self.client_id}...")
        loss = self.client.train_local(epochs=FL_CONFIG['local_epochs'])
        print(f"✓ Training complete - Loss: {loss:.4f}")
        return loss

    def upload_update(self, loss):
        """Upload model update to server"""
        try:
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

            print(f"✓ Uploaded update to server")
            return True

        except Exception as e:
            print(f"✗ Error uploading update: {e}")
            return False

    def run_training_loop(self, num_rounds=10):
        """Run federated learning training loop"""
        print("="*70)
        print(f"Starting Federated Training - Client {self.client_id}")
        print("="*70)

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            # Step 1: Download global model
            current_round = self.get_global_model()
            if current_round is None:
                print("Failed to get global model, retrying in 10s...")
                time.sleep(10)
                continue

            # Step 2: Train locally
            loss = self.train_local()

            # Step 3: Upload update
            success = self.upload_update(loss)
            if not success:
                print("Failed to upload update, retrying in 10s...")
                time.sleep(10)
                continue

            # Wait before next round
            print(f"Waiting for server aggregation...")
            time.sleep(20)  # Adjust based on your setup

        print("\n" + "="*70)
        print(f"Training complete for client {self.client_id}!")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', required=True, help='Unique client ID')
    parser.add_argument('--server-url', required=True, help='Server URL (e.g., http://192.168.1.100:8000)')
    parser.add_argument('--num-rounds', type=int, default=10, help='Number of training rounds')
    parser.add_argument('--data-size', type=int, default=1000, help='Number of local training samples')

    args = parser.parse_args()

    # Create local dataset
    print("Generating local training data...")
    from data.dataset import create_synthetic_data, MultimodalSecurityDataset

    local_data = create_synthetic_data(num_samples=args.data_size, threat_ratio=0.3)
    train_dataset = MultimodalSecurityDataset(local_data)

    # Create and run client
    client = RemoteFederatedClient(
        client_id=args.client_id,
        server_url=args.server_url,
        train_dataset=train_dataset
    )

    client.run_training_loop(num_rounds=args.num_rounds)


if __name__ == '__main__':
    main()
```

### Step 3: Install Dependencies

On **all machines** (server + clients):

```bash
pip install flask requests torch torchvision transformers
```

### Step 4: Deploy

**On Server Machine (192.168.1.100):**
```bash
python server_api.py --host 0.0.0.0 --port 8000
```

**On Client Machine 1 (192.168.1.101):**
```bash
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 10 \
    --data-size 1000
```

**On Client Machine 2 (192.168.1.102):**
```bash
python client_node.py \
    --client-id client_2 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 10 \
    --data-size 1000
```

**On Client Machine 3 (192.168.1.103):**
```bash
python client_node.py \
    --client-id client_3 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 10 \
    --data-size 1000
```

### Step 5: Trigger Aggregation

On server machine or via curl:
```bash
curl -X POST http://192.168.1.100:8000/trigger_aggregation
```

Or create an auto-aggregation script:
```bash
# auto_aggregate.sh
while true; do
    sleep 30
    curl -X POST http://192.168.1.100:8000/trigger_aggregation
done
```

---

## Method 2: gRPC

For better performance and streaming support.

### Step 1: Define Protocol

Create `federated_service.proto`:

```protobuf
syntax = "proto3";

package federated;

service FederatedService {
    rpc GetGlobalModel (ModelRequest) returns (ModelResponse);
    rpc UploadUpdate (UpdateRequest) returns (UpdateResponse);
    rpc GetStatus (StatusRequest) returns (StatusResponse);
}

message ModelRequest {
    string client_id = 1;
}

message ModelResponse {
    int32 round = 1;
    bytes model_parameters = 2;
}

message UpdateRequest {
    string client_id = 1;
    bytes model_parameters = 2;
    int32 num_samples = 3;
    float loss = 4;
}

message UpdateResponse {
    bool success = 1;
    string message = 2;
    int32 round = 3;
}

message StatusRequest {}

message StatusResponse {
    int32 current_round = 1;
    int32 pending_clients = 2;
}
```

### Step 2: Generate Code

```bash
pip install grpcio grpcio-tools

python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    federated_service.proto
```

### Step 3: Implement Server

```python
import grpc
from concurrent import futures
import federated_service_pb2
import federated_service_pb2_grpc

class FederatedServicer(federated_service_pb2_grpc.FederatedServiceServicer):
    def __init__(self):
        self.server = FederatedServer(...)

    def GetGlobalModel(self, request, context):
        params = self.server.get_global_parameters()
        serialized = pickle.dumps(params)
        return federated_service_pb2.ModelResponse(
            round=self.current_round,
            model_parameters=serialized
        )

    def UploadUpdate(self, request, context):
        # Handle update
        return federated_service_pb2.UpdateResponse(
            success=True,
            message="Update received"
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_service_pb2_grpc.add_FederatedServiceServicer_to_server(
        FederatedServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

---

## Method 3: Docker + Kubernetes

For production-grade deployment.

### Step 1: Create Dockerfiles

**Server Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "server_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

**Client Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "client_node.py"]
```

### Step 2: Build Images

```bash
# Build server
docker build -t dss-server:latest -f Dockerfile.server .

# Build client
docker build -t dss-client:latest -f Dockerfile.client .
```

### Step 3: Create Docker Compose

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  server:
    image: dss-server:latest
    ports:
      - "8000:8000"
    networks:
      - federated-net
    volumes:
      - ./models:/app/models
    environment:
      - NUM_ROUNDS=50

  client1:
    image: dss-client:latest
    depends_on:
      - server
    networks:
      - federated-net
    environment:
      - CLIENT_ID=client_1
      - SERVER_URL=http://server:8000
      - NUM_ROUNDS=50
    command: >
      python client_node.py
      --client-id client_1
      --server-url http://server:8000
      --num-rounds 50

  client2:
    image: dss-client:latest
    depends_on:
      - server
    networks:
      - federated-net
    environment:
      - CLIENT_ID=client_2
      - SERVER_URL=http://server:8000
      - NUM_ROUNDS=50
    command: >
      python client_node.py
      --client-id client_2
      --server-url http://server:8000
      --num-rounds 50

networks:
  federated-net:
    driver: bridge
```

### Step 4: Run

```bash
docker-compose up -d
```

### Step 5: Kubernetes Deployment

`k8s-deployment.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: federated-server
spec:
  selector:
    app: dss-server
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
  type: LoadBalancer

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: dss-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dss-server
  template:
    metadata:
      labels:
        app: dss-server
    spec:
      containers:
      - name: server
        image: dss-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: dss-clients
spec:
  replicas: 5
  selector:
    matchLabels:
      app: dss-client
  template:
    metadata:
      labels:
        app: dss-client
    spec:
      containers:
      - name: client
        image: dss-client:latest
        env:
        - name: SERVER_URL
          value: "http://federated-server:8000"
        - name: CLIENT_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## Security Considerations

### 1. TLS/SSL Encryption

**Enable HTTPS on server:**
```python
# server_api.py with SSL
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8443,
        ssl_context=('cert.pem', 'key.pem')
    )
```

Generate certificates:
```bash
openssl req -x509 -newkey rsa:4096 -nodes \
    -out cert.pem -keyout key.pem -days 365
```

### 2. Authentication

Add API key authentication:
```python
from functools import wraps
from flask import request, jsonify

API_KEYS = {
    'client_1': 'secret_key_1',
    'client_2': 'secret_key_2'
}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        client_id = request.headers.get('X-Client-ID')

        if not api_key or API_KEYS.get(client_id) != api_key:
            return jsonify({'error': 'Unauthorized'}), 401

        return f(*args, **kwargs)
    return decorated_function

@app.route('/upload_update', methods=['POST'])
@require_api_key
def upload_update():
    # ... existing code ...
```

Client usage:
```python
headers = {
    'X-API-Key': 'secret_key_1',
    'X-Client-ID': 'client_1'
}
response = requests.post(url, json=payload, headers=headers)
```

### 3. Firewall Rules

```bash
# On server (Ubuntu/Debian)
sudo ufw allow 8000/tcp
sudo ufw enable

# On AWS
# Security Group: Allow inbound TCP 8000 from client IPs only
```

### 4. VPN Setup

**Using Tailscale (Easiest):**
```bash
# On all machines
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Get tailscale IPs
tailscale ip -4
```

---

## Production Deployment

### AWS Deployment Example

**Architecture:**
```
┌─────────────────────────────────────┐
│  Application Load Balancer          │
│  (dss-lb.example.com)               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  EC2 Instance (Server)              │
│  - t3.xlarge (4 vCPU, 16GB)         │
│  - Region: us-east-1                │
└─────────────────────────────────────┘
               ▲
               │
      ┌────────┼────────┐
      │        │        │
┌─────▼──┐ ┌──▼────┐ ┌─▼──────┐
│Client 1│ │Client2│ │Client N│
│t3.large│ │Various│ │Various │
└────────┘ └───────┘ └────────┘
```

**Launch Server:**
```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxx \
    --user-data file://setup-server.sh

# setup-server.sh
#!/bin/bash
apt-get update
apt-get install -y python3-pip git
git clone https://github.com/your-repo/dss-project
cd dss-project
pip3 install -r requirements.txt
python3 server_api.py --host 0.0.0.0 --port 8000
```

**Launch Clients:**
```bash
# On each client machine
git clone https://github.com/your-repo/dss-project
cd dss-project
pip3 install -r requirements.txt

# Run client
python3 client_node.py \
    --client-id $(hostname) \
    --server-url http://dss-lb.example.com:8000 \
    --num-rounds 50
```

### Monitoring

**Add Prometheus metrics:**
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
rounds_completed = Counter('fl_rounds_completed', 'Completed rounds')
training_time = Histogram('fl_training_time', 'Training time per round')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

**Grafana Dashboard:**
- Track rounds completed
- Monitor client participation
- Plot loss curves
- Alert on failures

---

## Troubleshooting

### Common Issues

**1. Connection Refused**
```bash
# Check if server is running
curl http://server-ip:8000/health

# Check firewall
sudo ufw status
sudo iptables -L

# Check if port is open
netstat -tuln | grep 8000
```

**2. Model Serialization Errors**
```python
# Use consistent PyTorch versions
pip install torch==2.0.0  # Same version on all machines
```

**3. Slow Training**
```python
# Enable GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reduce batch size if OOM
FL_CONFIG['batch_size'] = 4

# Use mixed precision
torch.cuda.amp.autocast()
```

**4. Network Timeout**
```python
# Increase timeout
response = requests.post(url, json=payload, timeout=300)

# Add retry logic
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
```

---

## Summary

### Quick Start for 3 Machines

**Machine 1 (Server - 192.168.1.100):**
```bash
python server_api.py
```

**Machine 2 (Client 1 - 192.168.1.101):**
```bash
python client_node.py --client-id client_1 --server-url http://192.168.1.100:8000
```

**Machine 3 (Client 2 - 192.168.1.102):**
```bash
python client_node.py --client-id client_2 --server-url http://192.168.1.100:8000
```

**Trigger aggregation:**
```bash
curl -X POST http://192.168.1.100:8000/trigger_aggregation
```

### Best Practices

✅ Use HTTPS/TLS in production
✅ Implement authentication
✅ Set up monitoring and logging
✅ Use VPN for secure communication
✅ Regular backups of trained models
✅ Implement retry logic
✅ Use load balancing for scalability
✅ Document your network topology

---

## Next Steps

1. Test deployment on local network
2. Implement security (TLS, auth)
3. Add monitoring and alerts
4. Scale to more clients
5. Deploy to cloud (AWS/Azure/GCP)
6. Integrate with production systems

For more help, see:
- [README.md](README.md) - Project overview
- [PRESENTATION.md](PRESENTATION.md) - Detailed explanation
- GitHub Issues - Report problems
