# Quick Start: Multi-Machine Deployment

This guide will get you up and running with 3 machines in 10 minutes.

## Prerequisites

- 3 machines (physical or virtual) on the same network
- Python 3.8+ installed on all machines
- Network connectivity between machines

## Step-by-Step Setup

### 1. Identify Your Machines

Let's say you have:
- **Machine A** (Server): IP `192.168.1.100`
- **Machine B** (Client 1): IP `192.168.1.101`
- **Machine C** (Client 2): IP `192.168.1.102`

### 2. Install on All Machines

On **each machine**, run:

```bash
# Clone or copy the project
cd /path/to/project
git clone <your-repo> DSS_Project
cd DSS_Project

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Server (Machine A)

On **Machine A** (192.168.1.100):

```bash
python server_api.py --host 0.0.0.0 --port 8000
```

You should see:
```
======================================================================
Distributed Security Threat Detection System
Federated Learning Server
======================================================================
Server Address: 0.0.0.0:8000
Device: cpu
Max Rounds: 50
Clients per Round: 2
======================================================================

Waiting for clients to connect...
```

### 4. Start Client 1 (Machine B)

On **Machine B** (192.168.1.101):

```bash
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 10 \
    --data-size 500
```

### 5. Start Client 2 (Machine C)

On **Machine C** (192.168.1.102):

```bash
python client_node.py \
    --client-id client_2 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 10 \
    --data-size 500
```

### 6. Watch the Training

On both clients, you'll see output like:

```
======================================================================
Client client_1 Initialized
======================================================================
Server URL: http://192.168.1.100:8000
Device: cpu
Training samples: 500
======================================================================

✓ Server is online (Round 0/50)

──────────────────────────────────────────────────────────────────────
ROUND 1/10
──────────────────────────────────────────────────────────────────────
[10:30:15] ✓ Downloaded global model (Round 0)
[10:30:15] → Training on local data...
[10:30:45] ✓ Training complete | Loss: 0.0234 | Time: 30.1s
[10:30:45] → Uploading model update...
[10:30:47] ✓ Upload successful (Pending: 1)
[10:30:47] ⏳ Waiting 30s for aggregation...
```

On the server, you'll see:

```
[10:30:47] ✓ Received update from client_1 (Loss: 0.0234)
[10:30:48] ✓ Received update from client_2 (Loss: 0.0189)
[10:30:53] → Enough clients ready. Auto-aggregating...
[10:30:53] ✓ Round 1 complete | Clients: 2 | Avg Loss: 0.0211
```

## Verification

### Check Server Status

From any machine:
```bash
curl http://192.168.1.100:8000/health
```

Output:
```json
{
  "status": "online",
  "round": 1,
  "max_rounds": 50,
  "clients_registered": 0,
  "clients_required": 2,
  "device": "cpu"
}
```

### Check Training Progress

```bash
curl http://192.168.1.100:8000/get_status
```

Output:
```json
{
  "current_round": 1,
  "max_rounds": 50,
  "progress_percent": 2.0,
  "pending_clients": 0
}
```

## Troubleshooting

### "Cannot reach server"

**Problem:** Client can't connect to server

**Solution:**
```bash
# On server machine, check if server is running
curl http://localhost:8000/health

# Check firewall (Linux)
sudo ufw allow 8000/tcp

# Check firewall (Windows)
# Windows Defender Firewall > Allow an app > Python

# Verify server is listening
netstat -tuln | grep 8000
```

### "Connection timeout"

**Problem:** Network connectivity issues

**Solution:**
```bash
# Test basic connectivity
ping 192.168.1.100

# Test port connectivity
telnet 192.168.1.100 8000
# or
nc -zv 192.168.1.100 8000
```

### "Model serialization error"

**Problem:** PyTorch version mismatch

**Solution:**
```bash
# Ensure same PyTorch version on all machines
pip install torch==2.0.0 torchvision==0.15.0
```

## Advanced Options

### Run with GPU

If you have GPU:
```bash
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --device cuda
```

### More Training Rounds

```bash
python server_api.py --rounds 100 --clients-per-round 3
```

### Larger Dataset

```bash
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --data-size 5000
```

### Custom Wait Time

```bash
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --wait-time 60  # Wait 60 seconds between rounds
```

## Testing Locally (Single Machine)

You can test everything on one machine using different terminals:

**Terminal 1 (Server):**
```bash
python server_api.py
```

**Terminal 2 (Client 1):**
```bash
python client_node.py --client-id client_1 --server-url http://localhost:8000
```

**Terminal 3 (Client 2):**
```bash
python client_node.py --client-id client_2 --server-url http://localhost:8000
```

## Next Steps

Once basic deployment works:

1. ✅ Add more clients (3-10 machines)
2. ✅ Deploy across different networks (use VPN)
3. ✅ Enable HTTPS/TLS for security
4. ✅ Add authentication
5. ✅ Use real security data instead of synthetic
6. ✅ Deploy to cloud (AWS, Azure, GCP)
7. ✅ Set up monitoring and alerts

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed production deployment instructions.

## Common Commands Reference

### Server Commands

```bash
# Start server
python server_api.py

# Start server on specific port
python server_api.py --port 8080

# Configure training
python server_api.py --rounds 100 --clients-per-round 3

# Make server accessible from any IP
python server_api.py --host 0.0.0.0 --port 8000
```

### Client Commands

```bash
# Start client
python client_node.py --client-id <id> --server-url <url>

# Full configuration
python client_node.py \
    --client-id client_1 \
    --server-url http://192.168.1.100:8000 \
    --num-rounds 20 \
    --data-size 2000 \
    --device cuda \
    --wait-time 45
```

### API Commands (using curl)

```bash
# Health check
curl http://192.168.1.100:8000/health

# Get status
curl http://192.168.1.100:8000/get_status

# Manual aggregation trigger
curl -X POST http://192.168.1.100:8000/trigger_aggregation

# Save model
curl -X POST http://192.168.1.100:8000/save_model \
    -H "Content-Type: application/json" \
    -d '{"filepath": "my_model.pth"}'
```

## System Architecture

```
     Server (192.168.1.100:8000)
            ▲
            │ HTTP/REST
            │
     ┌──────┴──────┐
     │             │
Client 1      Client 2
(.101)        (.102)
```

## What Happens During Training?

1. **Initialization**
   - Server creates initial global model
   - Clients connect and register

2. **Each Round**
   - Clients download global model
   - Clients train on local data (3 epochs)
   - Clients upload model updates
   - Server aggregates updates (FedAvg)
   - Global model improves

3. **Completion**
   - After N rounds, training stops
   - Final model available on server
   - Clients can use model for inference

## Performance Expectations

### On CPU
- Training time per round: 30-60 seconds
- Detection time: ~500-1000ms per sample
- Suitable for: Testing, small deployments

### On GPU
- Training time per round: 5-15 seconds
- Detection time: ~3-10ms per sample
- Suitable for: Production, large scale

## Support

- **Documentation**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Presentation**: See [PRESENTATION.md](PRESENTATION.md)
- **Issues**: Create GitHub issue
- **Email**: [Your contact]

---

**That's it! You should now have a working distributed federated learning system running across multiple machines.**
