# Result.py

def calculate_final_verdict(blink_rate, voice_stress_prob, tension_consistency):
    """
    Each factor contributes 33.33% to the final score.
    blink_rate: total blinks in 1 minute
    voice_stress_prob: 0-100 score from voice.py
    tension_consistency: 0-100 percentage of frames marked 'Tense'
    """
    
    # 1. Blink Score (33.33%)
    # Normal is 15-20. We penalize if < 5 (hiding) or > 25 (anxiety)
    if blink_rate > 25 or blink_rate < 5:
        blink_score = 33.33
    elif blink_rate > 20:
        blink_score = 15.0
    else:
        blink_score = 0.0

    # 2. Voice Score (33.33%)
    # Convert the 0-100 voice prob to a max of 33.33
    voice_contribution = (voice_stress_prob / 100) * 33.33

    # 3. Facial Score (33.33%)
    # Convert the tension percentage to a max of 33.33
    facial_contribution = (tension_consistency / 100) * 33.33

    # Final Probability
    total_lie_probability = blink_score + voice_contribution + facial_contribution
    
    # Classification
    if total_lie_probability > 70:
        verdict = "DECEPTIVE (High Confidence)"
    elif total_lie_probability > 40:
        verdict = "UNCERTAIN / STRESSED"
    else:
        verdict = "TRUTHFUL"

    return round(total_lie_probability, 2), verdict
