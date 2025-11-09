"""
Federated Learning Server with REST API
Run this on the central server machine

Usage:
    python server_api.py --host 0.0.0.0 --port 8000
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
from datetime import datetime

app = Flask(__name__)

# Global server instance
global_model = MultimodalFusionModel(MODEL_CONFIG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
server = FederatedServer(global_model, device=device)

# Store client updates
client_updates = {}
update_lock = threading.Lock()

# Training state
current_round = 0
max_rounds = 50
clients_per_round = 2
start_time = time.time()


@app.route('/', methods=['GET'])
def home():
    """Home page with API information"""
    return jsonify({
        'service': 'Federated Learning Server',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'get_model': '/get_global_model',
            'upload': '/upload_update',
            'aggregate': '/trigger_aggregation',
            'status': '/get_status'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is alive"""
    uptime = time.time() - start_time
    return jsonify({
        'status': 'online',
        'round': current_round,
        'max_rounds': max_rounds,
        'clients_registered': len(client_updates),
        'clients_required': clients_per_round,
        'uptime_seconds': uptime,
        'device': device,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """
    Clients request the current global model
    Returns: Serialized model parameters
    """
    try:
        global_params = server.get_global_parameters()

        # Serialize parameters
        serialized = pickle.dumps(global_params)
        encoded = base64.b64encode(serialized).decode('utf-8')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model downloaded by client")

        return jsonify({
            'status': 'success',
            'round': current_round,
            'model_parameters': encoded,
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


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
    try:
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

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ✓ Received update from {client_id} (Loss: {data['loss']:.4f})")

        # Check if we can auto-aggregate
        if len(client_updates) >= clients_per_round:
            print(f"[{timestamp}] → Enough clients ready. Auto-aggregating...")
            threading.Thread(target=auto_aggregate).start()

        return jsonify({
            'status': 'success',
            'message': f'Update received from {client_id}',
            'round': current_round,
            'pending_clients': len(client_updates)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def auto_aggregate():
    """Automatically aggregate when enough clients are ready"""
    global current_round

    time.sleep(5)  # Wait a bit for any stragglers

    with update_lock:
        if len(client_updates) < clients_per_round:
            return

        try:
            # Extract parameters and weights
            client_params = [u['parameters'] for u in client_updates.values()]
            client_weights = [u['num_samples'] for u in client_updates.values()]

            # Aggregate
            server.update_global_model(client_params, client_weights)
            current_round += 1

            avg_loss = sum(u['loss'] for u in client_updates.values()) / len(client_updates)

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ✓ Round {current_round} complete | "
                  f"Clients: {len(client_updates)} | Avg Loss: {avg_loss:.4f}")

            # Clear updates for next round
            client_updates.clear()

        except Exception as e:
            print(f"Error during aggregation: {e}")


@app.route('/trigger_aggregation', methods=['POST'])
def trigger_aggregation():
    """
    Manually trigger aggregation when enough clients have uploaded
    """
    global current_round

    with update_lock:
        if len(client_updates) < clients_per_round:
            return jsonify({
                'status': 'waiting',
                'message': f'Only {len(client_updates)}/{clients_per_round} clients ready'
            }), 400

        try:
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
                'avg_loss': avg_loss,
                'timestamp': datetime.now().isoformat()
            }

            client_updates.clear()

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ✓ Round {current_round} complete | Avg Loss: {avg_loss:.4f}")

            return jsonify(result)

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500


@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current training status"""
    with update_lock:
        pending = list(client_updates.keys())

    return jsonify({
        'current_round': current_round,
        'max_rounds': max_rounds,
        'pending_clients': len(pending),
        'pending_client_ids': pending,
        'clients_per_round': clients_per_round,
        'progress_percent': (current_round / max_rounds) * 100,
        'device': device
    })


@app.route('/save_model', methods=['POST'])
def save_model():
    """Save the current global model"""
    try:
        filepath = request.json.get('filepath', 'global_model.pth')
        torch.save(server.global_model.state_dict(), filepath)
        return jsonify({
            'status': 'success',
            'message': f'Model saved to {filepath}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def run_server(host='0.0.0.0', port=8000):
    """Start the federated learning server"""
    print("="*70)
    print("Distributed Security Threat Detection System")
    print("Federated Learning Server")
    print("="*70)
    print(f"Server Address: {host}:{port}")
    print(f"Device: {device}")
    print(f"Max Rounds: {max_rounds}")
    print(f"Clients per Round: {clients_per_round}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\nWaiting for clients to connect...")
    print("\nAPI Endpoints:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  GET  http://{host}:{port}/get_global_model")
    print(f"  POST http://{host}:{port}/upload_update")
    print(f"  POST http://{host}:{port}/trigger_aggregation")
    print(f"  GET  http://{host}:{port}/get_status")
    print("="*70)
    print()

    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host address')
    parser.add_argument('--port', default=8000, type=int, help='Server port')
    parser.add_argument('--rounds', default=50, type=int, help='Maximum training rounds')
    parser.add_argument('--clients-per-round', default=2, type=int, help='Clients per round')

    args = parser.parse_args()

    max_rounds = args.rounds
    clients_per_round = args.clients_per_round

    run_server(host=args.host, port=args.port)
