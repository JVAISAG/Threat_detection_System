"""
SUPER SIMPLE VERSION - For Complete Beginners
Only ~100 lines, easy to understand

What this does:
- Detects if network activity is a threat or normal
- Uses a simple scoring system (no complex ML needed for understanding)
- Shows real-time detection

Run: python super_simple.py
"""

import random


# ============================================================================
# SIMPLE THREAT DETECTION (No fancy ML, just math!)
# ============================================================================

class SimpleDetector:
    """Detects threats using simple rules"""

    def __init__(self):
        self.threshold = 5.0  # If score > 5, it's a threat

    def calculate_threat_score(self, activity):
        """Calculate how suspicious the activity is"""
        score = 0

        # Check for suspicious keywords
        suspicious_words = ['injection', 'attack', 'failed', 'unauthorized', 'exploit']
        for word in suspicious_words:
            if word in activity.lower():
                score += 2

        # Check for high numbers (could indicate DDoS)
        if '1000' in activity or '10000' in activity:
            score += 3

        # Check for IP addresses (could be scanning)
        if activity.count('.') >= 3:  # Possible IP address
            score += 1

        return score

    def is_threat(self, activity):
        """Decide if activity is a threat"""
        score = self.calculate_threat_score(activity)
        return score >= self.threshold, score


# ============================================================================
# SIMULATE ACTIVITIES
# ============================================================================

def get_sample_activities():
    """Get sample network activities to test"""
    return [
        # Normal activities
        ("User login successful from 192.168.1.100", False),
        ("Email sent successfully", False),
        ("File backup completed", False),
        ("Database query executed", False),

        # Threats
        ("SQL injection attempt detected", True),
        ("10000 failed login attempts", True),
        ("Unauthorized access from 10.0.0.50", True),
        ("DDoS attack from 185.220.101.1", True),
        ("Port scanning detected on 65535 ports", True),
    ]


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("SUPER SIMPLE THREAT DETECTOR")
    print("=" * 70)
    print()
    print("How it works:")
    print("  1. Reads network activity (logs)")
    print("  2. Calculates threat score")
    print("  3. If score > 5 → THREAT, else → SAFE")
    print()
    print("=" * 70)
    print()

    # Create detector
    detector = SimpleDetector()

    # Get test activities
    activities = get_sample_activities()

    # Test each activity
    correct = 0
    total = len(activities)

    for activity, is_actual_threat in activities:
        # Detect
        is_detected_threat, score = detector.is_threat(activity)

        # Check if correct
        is_correct = (is_detected_threat == is_actual_threat)
        if is_correct:
            correct += 1

        # Display result
        print(f"Activity: {activity}")
        print(f"  Threat Score: {score}")
        print(f"  Expected: {'THREAT' if is_actual_threat else 'SAFE'}")
        print(f"  Detected: {'THREAT' if is_detected_threat else 'SAFE'}")

        if is_correct:
            print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ WRONG")
        print()

    # Summary
    accuracy = (correct / total) * 100
    print("=" * 70)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.0f}% accuracy)")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Try: python simple_demo.py (uses actual neural network)")
    print("  • Try: python demo_threat_detection.py (full system)")
    print()


if __name__ == '__main__':
    main()
