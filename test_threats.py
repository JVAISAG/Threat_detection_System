"""
Threat Simulation and Detection Testing
Simulate various cyberattacks and test the trained model's detection capabilities

Usage:
    python test_threats.py --model-path global_model.pth
"""

import torch
import numpy as np
import argparse
from models.multimodal_model import MultimodalFusionModel
from detection.threat_detector import ThreatDetector
from config import MODEL_CONFIG
import time


class ThreatSimulator:
    """Simulate various types of cyber threats for testing"""

    def __init__(self):
        self.threat_types = [
            'sql_injection',
            'ddos_attack',
            'brute_force',
            'port_scanning',
            'malware_detected',
            'data_exfiltration',
            'privilege_escalation',
            'zero_day_exploit',
            'ransomware',
            'phishing'
        ]

    def generate_sql_injection(self):
        """Simulate SQL injection attack"""
        logs = [
            "SQL injection attempt: SELECT * FROM users WHERE id='1' OR '1'='1'",
            "Suspicious SQL query detected: DROP TABLE users; --",
            "SQL syntax error in user input: ' OR 1=1--",
            "Malformed query: UNION SELECT password FROM admin_users",
            "SQL injection vector: admin' AND 1=1 UNION SELECT NULL--"
        ]

        # Abnormal network pattern for SQL injection
        sensor_data = np.random.randn(100) + 3.0  # High anomaly score
        sensor_data[10:20] += 2.0  # Spike in specific features

        return {
            'threat_type': 'SQL Injection',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'CRITICAL'
        }

    def generate_ddos_attack(self):
        """Simulate DDoS attack"""
        logs = [
            "Abnormal traffic: 10000 requests from IP 185.220.101.50 in 1 second",
            "DDoS attack detected: SYN flood from multiple IPs",
            "Traffic spike: 500% increase in HTTP requests",
            "Rate limit exceeded: 1000+ requests per second",
            "Botnet activity detected: coordinated traffic from 200 IPs"
        ]

        # Extreme traffic pattern
        sensor_data = np.random.randn(100) + 5.0  # Very high anomaly
        sensor_data[0:30] += 3.0  # Massive spike in traffic metrics

        return {
            'threat_type': 'DDoS Attack',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'CRITICAL'
        }

    def generate_brute_force(self):
        """Simulate brute force attack"""
        logs = [
            "Failed login attempts: 50 failed logins for user 'admin' from IP 10.0.0.50",
            "Multiple authentication failures from 192.168.1.100",
            "Suspicious activity: 100 password attempts in 5 minutes",
            "Brute force detected: dictionary attack on SSH service",
            "Account locked: 20 failed login attempts for user 'root'"
        ]

        # Elevated authentication metrics
        sensor_data = np.random.randn(100) + 2.5
        sensor_data[40:60] += 2.0  # Spike in auth attempts

        return {
            'threat_type': 'Brute Force',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'HIGH'
        }

    def generate_port_scanning(self):
        """Simulate port scanning"""
        logs = [
            "Port scan detected: 65535 ports scanned from IP 192.168.1.200",
            "Nmap scan detected: SYN scan on all TCP ports",
            "Suspicious activity: sequential port probing detected",
            "Port enumeration: scanning ports 1-1024 from external IP",
            "Stealth scan detected: FIN scan on common services"
        ]

        # Port scanning signature
        sensor_data = np.random.randn(100) + 2.0
        sensor_data[60:80] += 1.5  # Port diversity spike

        return {
            'threat_type': 'Port Scanning',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'MEDIUM'
        }

    def generate_malware(self):
        """Simulate malware detection"""
        logs = [
            "Malware detected: Trojan.Win32.Generic found in system32",
            "Virus signature match: wannacry.exe detected",
            "Suspicious process: unknown binary accessing registry",
            "Ransomware activity: mass file encryption detected",
            "Rootkit detected: hidden processes found"
        ]

        # Malware behavior pattern
        sensor_data = np.random.randn(100) + 3.5
        sensor_data[20:40] += 2.5  # Process/file activity spike

        return {
            'threat_type': 'Malware',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'CRITICAL'
        }

    def generate_data_exfiltration(self):
        """Simulate data exfiltration"""
        logs = [
            "Data exfiltration: 10GB uploaded to unknown external IP",
            "Suspicious outbound traffic: large file transfer detected",
            "Unusual data transfer: database dump sent to 185.220.101.1",
            "Data leak detected: sensitive files uploaded to cloud storage",
            "Unauthorized data access: bulk download of customer records"
        ]

        # Data transfer anomaly
        sensor_data = np.random.randn(100) + 2.8
        sensor_data[0:20] += 3.0  # Outbound traffic spike

        return {
            'threat_type': 'Data Exfiltration',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'CRITICAL'
        }

    def generate_privilege_escalation(self):
        """Simulate privilege escalation"""
        logs = [
            "Privilege escalation: user 'guest' gained admin rights",
            "Unauthorized sudo access: privilege elevation detected",
            "Suspicious activity: normal user executing root commands",
            "Exploit detected: CVE-2021-3156 sudo vulnerability",
            "Account privilege changed: user added to administrators group"
        ]

        # Privilege anomaly
        sensor_data = np.random.randn(100) + 2.2
        sensor_data[30:50] += 1.8

        return {
            'threat_type': 'Privilege Escalation',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'HIGH'
        }

    def generate_zero_day(self):
        """Simulate zero-day exploit"""
        logs = [
            "Unknown exploit detected: unusual process behavior",
            "Zero-day vulnerability exploited: CVE-2024-XXXX",
            "Suspicious activity: unknown attack vector detected",
            "Anomalous behavior: exploit attempt on unpatched service",
            "Critical alert: potential zero-day exploitation"
        ]

        # Unknown threat pattern
        sensor_data = np.random.randn(100) + 4.0
        sensor_data += np.random.randn(100) * 0.5  # More random/chaotic

        return {
            'threat_type': 'Zero-Day Exploit',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'CRITICAL'
        }

    def generate_normal_activity(self):
        """Simulate normal system activity"""
        logs = [
            "User login successful from IP 192.168.1.100",
            "System backup completed successfully",
            "Database connection established",
            "File access granted to authorized user",
            "Network traffic within normal parameters",
            "Scheduled maintenance completed",
            "User 'john.doe' logged in from workstation-42",
            "Email sent successfully",
            "Web server responding normally",
            "Routine security scan completed"
        ]

        # Normal network pattern
        sensor_data = np.random.randn(100) * 0.5  # Low variance, centered at 0

        return {
            'threat_type': 'Normal Activity',
            'log': np.random.choice(logs),
            'sensor': sensor_data,
            'severity': 'NONE'
        }

    def generate_threat(self, threat_type=None):
        """Generate a specific threat or random one"""
        threat_generators = {
            'sql_injection': self.generate_sql_injection,
            'ddos_attack': self.generate_ddos_attack,
            'brute_force': self.generate_brute_force,
            'port_scanning': self.generate_port_scanning,
            'malware_detected': self.generate_malware,
            'data_exfiltration': self.generate_data_exfiltration,
            'privilege_escalation': self.generate_privilege_escalation,
            'zero_day_exploit': self.generate_zero_day,
            'normal': self.generate_normal_activity
        }

        if threat_type is None:
            threat_type = np.random.choice(list(threat_generators.keys()))

        return threat_generators[threat_type]()


def test_threat_detection(model_path=None, num_tests=20):
    """Test the model with various simulated threats"""

    print("="*70)
    print("THREAT DETECTION TESTING")
    print("="*70)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    model = MultimodalFusionModel(MODEL_CONFIG)

    # Load trained model if provided
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded trained model from {model_path}")
        except:
            print(f"⚠️ Could not load model from {model_path}, using untrained model")
    else:
        print("⚠️ No model path provided, using untrained model")

    # Initialize detector
    detector = ThreatDetector(model, device=device, threshold=0.5)
    simulator = ThreatSimulator()

    # Test results storage
    results = {
        'total': 0,
        'correct': 0,
        'threats_detected': 0,
        'threats_missed': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'by_type': {}
    }

    print(f"\nRunning {num_tests} threat detection tests...")
    print("="*70)

    # Run tests
    for i in range(num_tests):
        # Mix of threats and normal activity
        if i < num_tests * 0.7:  # 70% threats
            threat_types = ['sql_injection', 'ddos_attack', 'brute_force',
                          'port_scanning', 'malware_detected', 'data_exfiltration',
                          'privilege_escalation', 'zero_day_exploit']
            threat = simulator.generate_threat(np.random.choice(threat_types))
            is_actual_threat = True
        else:  # 30% normal
            threat = simulator.generate_normal_activity()
            is_actual_threat = False

        # Detect
        start_time = time.time()
        result = detector.detect_threat(
            text=threat['log'],
            sensor_data=threat['sensor']
        )
        detection_time = time.time() - start_time

        # Evaluate
        is_detected_threat = result['is_threat']
        results['total'] += 1

        # Update statistics
        threat_type = threat['threat_type']
        if threat_type not in results['by_type']:
            results['by_type'][threat_type] = {
                'total': 0,
                'detected': 0,
                'missed': 0
            }

        results['by_type'][threat_type]['total'] += 1

        # Determine correctness
        correct = (is_actual_threat == is_detected_threat)
        if correct:
            results['correct'] += 1

        if is_actual_threat and is_detected_threat:
            # True Positive
            results['threats_detected'] += 1
            results['by_type'][threat_type]['detected'] += 1
            status = "✓ DETECTED"
            color = '\033[92m'  # Green
        elif is_actual_threat and not is_detected_threat:
            # False Negative
            results['threats_missed'] += 1
            results['by_type'][threat_type]['missed'] += 1
            status = "✗ MISSED"
            color = '\033[91m'  # Red
        elif not is_actual_threat and is_detected_threat:
            # False Positive
            results['false_positives'] += 1
            status = "⚠ FALSE ALARM"
            color = '\033[93m'  # Yellow
        else:
            # True Negative
            results['true_negatives'] += 1
            status = "✓ CORRECT (Normal)"
            color = '\033[92m'  # Green

        # Print result
        print(f"\nTest {i+1}/{num_tests}:")
        print(f"  Type: {threat_type}")
        print(f"  Log: {threat['log'][:70]}...")
        print(f"  Expected: {'THREAT' if is_actual_threat else 'NORMAL'}")
        print(f"  Detected: {'THREAT' if is_detected_threat else 'NORMAL'}")
        print(f"  Probability: {result['threat_probability']:.2%}")
        print(f"  Level: {result['threat_level']}")
        print(f"  Time: {detection_time*1000:.2f}ms")
        print(f"  {color}Status: {status}\033[0m")

    # Print summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)

    accuracy = (results['correct'] / results['total']) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Total Tests: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['total'] - results['correct']}")

    print(f"\nThreat Detection:")
    actual_threats = results['threats_detected'] + results['threats_missed']
    if actual_threats > 0:
        detection_rate = (results['threats_detected'] / actual_threats) * 100
        print(f"  Detected: {results['threats_detected']}/{actual_threats} ({detection_rate:.1f}%)")
        print(f"  Missed: {results['threats_missed']}/{actual_threats}")

    print(f"\nFalse Alarms:")
    actual_normal = results['true_negatives'] + results['false_positives']
    if actual_normal > 0:
        fpr = (results['false_positives'] / actual_normal) * 100
        print(f"  False Positives: {results['false_positives']}/{actual_normal} ({fpr:.1f}%)")
        print(f"  True Negatives: {results['true_negatives']}/{actual_normal}")

    print(f"\n" + "="*70)
    print("DETECTION BY THREAT TYPE")
    print("="*70)

    for threat_type, stats in sorted(results['by_type'].items()):
        if stats['total'] > 0:
            detection_rate = (stats['detected'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"\n{threat_type}:")
            print(f"  Total: {stats['total']}")
            print(f"  Detected: {stats['detected']} ({detection_rate:.1f}%)")
            if stats['missed'] > 0:
                print(f"  Missed: {stats['missed']}")

    print("\n" + "="*70)

    return results


def interactive_threat_test():
    """Interactive mode to test specific threats"""

    print("="*70)
    print("INTERACTIVE THREAT TESTING")
    print("="*70)

    # Initialize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalFusionModel(MODEL_CONFIG)
    detector = ThreatDetector(model, device=device, threshold=0.5)
    simulator = ThreatSimulator()

    print("\nAvailable threat types:")
    threat_types = [
        '1. SQL Injection',
        '2. DDoS Attack',
        '3. Brute Force',
        '4. Port Scanning',
        '5. Malware',
        '6. Data Exfiltration',
        '7. Privilege Escalation',
        '8. Zero-Day Exploit',
        '9. Normal Activity'
    ]

    for t in threat_types:
        print(f"  {t}")

    threat_map = {
        '1': 'sql_injection',
        '2': 'ddos_attack',
        '3': 'brute_force',
        '4': 'port_scanning',
        '5': 'malware_detected',
        '6': 'data_exfiltration',
        '7': 'privilege_escalation',
        '8': 'zero_day_exploit',
        '9': 'normal'
    }

    while True:
        print("\n" + "="*70)
        choice = input("\nSelect threat type (1-9) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            break

        if choice not in threat_map:
            print("Invalid choice!")
            continue

        # Generate threat
        threat = simulator.generate_threat(threat_map[choice])

        print("\n" + "-"*70)
        print(f"SIMULATED: {threat['threat_type']}")
        print("-"*70)
        print(f"Log: {threat['log']}")
        print(f"Severity: {threat['severity']}")

        # Detect
        result = detector.detect_threat(
            text=threat['log'],
            sensor_data=threat['sensor']
        )

        print("\n" + "-"*70)
        print("DETECTION RESULT")
        print("-"*70)
        print(f"Threat Detected: {'YES' if result['is_threat'] else 'NO'}")
        print(f"Threat Probability: {result['threat_probability']:.2%}")
        print(f"Threat Level: {result['threat_level']}")
        print(f"Detection Time: {result['detection_time_ms']:.2f}ms")

        if result['is_threat']:
            print("\n⚠️  ALERT: Threat detected! Immediate action required.")
        else:
            print("\n✓ System appears normal.")


def main():
    parser = argparse.ArgumentParser(description='Threat Detection Testing')
    parser.add_argument('--model-path', help='Path to trained model (.pth file)')
    parser.add_argument('--num-tests', type=int, default=20, help='Number of tests to run')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')

    args = parser.parse_args()

    if args.interactive:
        interactive_threat_test()
    else:
        test_threat_detection(model_path=args.model_path, num_tests=args.num_tests)


if __name__ == '__main__':
    main()
