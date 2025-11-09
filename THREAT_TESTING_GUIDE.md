# Threat Simulation & Detection Testing Guide

This guide explains how to simulate cyber threats and test if your trained model can detect them.

## Overview

Your system can now:
- ✅ **Simulate** 8+ types of cyber attacks
- ✅ **Detect** threats in real-time
- ✅ **Test** model accuracy with various threats
- ✅ **Demonstrate** specific attack scenarios

---

## Quick Start

### 1. Simple Demo (Recommended First)

Run the complete demo that trains and tests:

```bash
python demo_threat_detection.py
```

This will:
1. Train a model (3 rounds, ~2 minutes)
2. Test with 10 simulated threats
3. Demonstrate specific attack detections

**Expected Output:**
```
======================================================================
THREAT DETECTION DEMONSTRATION
======================================================================

STEP 1: TRAINING MODEL
Training on: cpu
Rounds: 3
✓ Created 500 training samples

Round 1/3
  Loss: 0.0234

STEP 2: TESTING WITH SIMULATED THREATS

Test 1/10 🚨
  Type: SQL Injection
  Log: SQL injection attempt: SELECT * FROM users WHERE id='1' OR...
  Expected: THREAT
  Detected: THREAT (98.5%)
  Result: ✓ THREAT DETECTED
```

### 2. Automated Testing

Test with many threats:

```bash
# Test with 50 simulated threats
python test_threats.py --num-tests 50

# Test with trained model
python test_threats.py --model-path global_model.pth --num-tests 50
```

### 3. Interactive Testing

Test specific threats manually:

```bash
python test_threats.py --interactive
```

Choose threats interactively and see detection results.

---

## Simulated Threat Types

Your system can simulate these attacks:

| # | Threat Type | Severity | Description |
|---|-------------|----------|-------------|
| 1 | **SQL Injection** | CRITICAL | Database query manipulation |
| 2 | **DDoS Attack** | CRITICAL | Distributed denial of service |
| 3 | **Brute Force** | HIGH | Password cracking attempts |
| 4 | **Port Scanning** | MEDIUM | Network reconnaissance |
| 5 | **Malware** | CRITICAL | Virus/trojan/ransomware |
| 6 | **Data Exfiltration** | CRITICAL | Unauthorized data theft |
| 7 | **Privilege Escalation** | HIGH | Unauthorized access elevation |
| 8 | **Zero-Day Exploit** | CRITICAL | Unknown vulnerability exploit |
| 9 | **Normal Activity** | NONE | Legitimate system operations |

---

## How Threats are Simulated

### Example 1: SQL Injection

**Simulated Log:**
```
SQL injection attempt: SELECT * FROM users WHERE id='1' OR '1'='1'
```

**Simulated Network Pattern:**
```python
# Abnormal database query pattern
sensor_data = normal_pattern + anomaly_signature
# High spike in specific features
```

**Detection Result:**
```
✓ THREAT DETECTED
Probability: 99.2%
Level: CRITICAL
Time: 3.4ms
```

### Example 2: DDoS Attack

**Simulated Log:**
```
Abnormal traffic: 10000 requests from IP 185.220.101.50 in 1 second
```

**Simulated Network Pattern:**
```python
# Extreme traffic spike
sensor_data = normal + 5.0  # Very high anomaly
# Massive concurrent connections
```

**Detection Result:**
```
✓ THREAT DETECTED
Probability: 100.0%
Level: CRITICAL
Time: 2.8ms
```

### Example 3: Normal Activity

**Simulated Log:**
```
User login successful from IP 192.168.1.100
```

**Simulated Network Pattern:**
```python
# Normal baseline pattern
sensor_data = np.random.randn(100) * 0.5
# Low variance, centered at zero
```

**Detection Result:**
```
✓ NORMAL (Correct)
Probability: 2.1%
Level: LOW
Time: 3.1ms
```

---

## Testing Workflows

### Workflow 1: Basic Testing

```bash
# Step 1: Train model (if not already trained)
python main_lite.py

# Step 2: Test with threats
python test_threats.py --num-tests 20
```

### Workflow 2: With Pre-trained Model

```bash
# Step 1: Train and save model
python main_lite.py
# Model is saved automatically

# Step 2: Load and test
python test_threats.py --model-path global_model.pth --num-tests 50
```

### Workflow 3: Interactive Exploration

```bash
# Start interactive mode
python test_threats.py --interactive

# Select threats:
# 1. SQL Injection
# 2. DDoS Attack
# ... etc
```

### Workflow 4: Custom Threat Creation

Create your own threats in Python:

```python
from test_threats import ThreatSimulator
from detection.threat_detector import ThreatDetector
from models.multimodal_model import MultimodalFusionModel

# Initialize
simulator = ThreatSimulator()
model = MultimodalFusionModel(MODEL_CONFIG)
detector = ThreatDetector(model)

# Generate custom threat
threat = {
    'log': 'Your custom security log here',
    'sensor': your_network_traffic_features  # shape: (100,)
}

# Detect
result = detector.detect_threat(
    text=threat['log'],
    sensor_data=threat['sensor']
)

print(f"Threat: {result['is_threat']}")
print(f"Probability: {result['threat_probability']:.2%}")
```

---

## Understanding Detection Results

### Output Fields

```python
result = {
    'is_threat': True/False,           # Binary decision
    'threat_probability': 0.0-1.0,     # Confidence score
    'normal_probability': 0.0-1.0,     # Opposite of threat_probability
    'confidence': 0.0-1.0,             # max(threat, normal)
    'detection_time_ms': float,        # Processing time
    'threat_level': 'LOW|MEDIUM|HIGH|CRITICAL'  # Severity
}
```

### Threat Levels

| Probability | Threat Level | Action |
|-------------|--------------|---------|
| < 30% | LOW | Monitor |
| 30-60% | MEDIUM | Investigate |
| 60-80% | HIGH | Alert team |
| > 80% | CRITICAL | Immediate response |

### Interpreting Accuracy

```
Accuracy = (Correct Detections) / (Total Tests) × 100%

Where Correct = True Positives + True Negatives
```

**Example:**
```
Total Tests: 20
Threats Detected: 13/14 (92.9%)
Normal Identified: 5/6 (83.3%)
Overall Accuracy: 18/20 (90%)
```

---

## Performance Benchmarks

### Expected Results (After Training)

| Metric | Untrained | After 5 Rounds | After 50 Rounds |
|--------|-----------|----------------|-----------------|
| **Accuracy** | ~50% (random) | ~85-90% | ~95-100% |
| **Detection Rate** | ~50% | ~90-95% | ~98-100% |
| **False Positives** | ~50% | ~10-15% | ~0-5% |
| **Detection Time** | 3-5ms | 3-5ms | 3-5ms |

### Factors Affecting Accuracy

✅ **Improves with:**
- More training rounds
- Larger training dataset
- Better quality data
- Balanced threat/normal ratio

❌ **Degrades with:**
- Too little training
- Imbalanced data
- Unknown threat types (zero-days)
- Hardware issues

---

## Advanced Testing

### Test on Real Data

Replace synthetic data with real security logs:

```python
from test_threats import ThreatSimulator

# Your real security event
real_threat = {
    'log': "Actual security log from your system",
    'sensor': actual_network_metrics  # Extract from your monitoring
}

result = detector.detect_threat(
    text=real_threat['log'],
    sensor_data=real_threat['sensor']
)
```

### Batch Testing

Test multiple threats at once:

```python
simulator = ThreatSimulator()

# Generate 100 threats
threats = [simulator.generate_threat() for _ in range(100)]

# Batch detect
results = detector.batch_detect(threats)

# Analyze
accuracy = sum(1 for r in results if r['is_threat']) / len(results)
```

### A/B Testing

Compare different models:

```python
# Model A (5 rounds training)
model_a = load_model('model_5_rounds.pth')
detector_a = ThreatDetector(model_a)

# Model B (50 rounds training)
model_b = load_model('model_50_rounds.pth')
detector_b = ThreatDetector(model_b)

# Test same threats on both
for threat in test_threats:
    result_a = detector_a.detect_threat(...)
    result_b = detector_b.detect_threat(...)
    # Compare results
```

---

## Visualization & Reporting

### Generate Test Report

```python
# Run tests
results = test_threat_detection(num_tests=100)

# Save report
with open('detection_report.txt', 'w') as f:
    f.write(f"Accuracy: {results['accuracy']}%\n")
    f.write(f"Threats Detected: {results['threats_detected']}\n")
    f.write(f"False Positives: {results['false_positives']}\n")
```

### Plot Detection Performance

```python
import matplotlib.pyplot as plt

# Detection rates by threat type
threat_types = results['by_type'].keys()
detection_rates = [v['detected']/v['total'] for v in results['by_type'].values()]

plt.bar(threat_types, detection_rates)
plt.title('Detection Rate by Threat Type')
plt.ylabel('Detection Rate (%)')
plt.xticks(rotation=45)
plt.savefig('detection_rates.png')
```

---

## Troubleshooting

### Issue 1: Low Accuracy

**Problem:** Model detects < 70% of threats

**Solutions:**
```bash
# Train longer
python main_lite.py  # Let it run more rounds

# Use more data
# Edit data/dataset.py to generate more samples

# Check model is trained
# Make sure you're not using untrained model
```

### Issue 2: High False Positives

**Problem:** Normal activity flagged as threats

**Solutions:**
```python
# Adjust threshold
detector = ThreatDetector(model, threshold=0.7)  # More conservative

# Retrain with more normal examples
# Increase normal activity ratio in training data
```

### Issue 3: Detection Too Slow

**Problem:** Detection time > 100ms

**Solutions:**
```bash
# Use GPU
--device cuda

# Reduce batch size
# Edit config.py

# Use smaller model
# Reduce MODEL_CONFIG dimensions
```

---

## Real-world Integration

### Integrate with SIEM

```python
# Example: Splunk integration
def check_splunk_event(event):
    # Extract log and metrics
    log_text = event['log_message']
    network_metrics = extract_metrics(event)

    # Detect
    result = detector.detect_threat(
        text=log_text,
        sensor_data=network_metrics
    )

    # Alert if threat
    if result['is_threat'] and result['threat_level'] == 'CRITICAL':
        send_alert(event, result)
```

### Continuous Monitoring

```python
# Monitor in real-time
import time

while True:
    # Get latest security events
    events = fetch_security_events()

    for event in events:
        result = detector.detect_threat(
            text=event['log'],
            sensor_data=event['metrics']
        )

        if result['is_threat']:
            log_alert(event, result)
            notify_security_team(event, result)

    time.sleep(60)  # Check every minute
```

---

## Summary

### What You Can Do Now

✅ **Simulate** 8+ types of cyber attacks
✅ **Test** model with automated threat generation
✅ **Detect** threats in real-time with probability scores
✅ **Evaluate** model accuracy and performance
✅ **Demonstrate** specific attack scenarios
✅ **Integrate** with real security systems

### Quick Commands Reference

```bash
# Simple demo
python demo_threat_detection.py

# Automated testing
python test_threats.py --num-tests 50

# Interactive testing
python test_threats.py --interactive

# With trained model
python test_threats.py --model-path global_model.pth --num-tests 100
```

### Next Steps

1. ✅ Run `demo_threat_detection.py` to see it work
2. ✅ Test with `test_threats.py --num-tests 50`
3. ✅ Train longer with `main_lite.py` for better accuracy
4. ✅ Integrate with your real security data
5. ✅ Deploy to production for continuous monitoring

---

## Additional Resources

- **[README.md](README.md)** - Project overview
- **[QUICK_START.md](QUICK_START.md)** - Multi-machine deployment
- **[PRESENTATION.md](PRESENTATION.md)** - Detailed explanation
- **[test_threats.py](test_threats.py)** - Threat simulation code
- **[demo_threat_detection.py](demo_threat_detection.py)** - Demo script

---

**Ready to detect threats? Start with:** `python demo_threat_detection.py`
