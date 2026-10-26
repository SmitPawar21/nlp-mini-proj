import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import UnifiedPredictor

def run_tests():
    p = UnifiedPredictor()

    test_cases = [
        # Safety & Distress
        ("Sanchit ko marunga", "threat_violence"),
        ("Mujhe marna", "mental_health_distress"),
        ("mujhe marna hai jeene ka man nahi karta", "mental_health_distress"),
        ("main use peet dunga ache se", "threat_violence"),
        ("police ko call lagao accident ho gaya emergency", "emergency_help"),

        # Core Utilities
        ("Delhi se Mumbai flight book karo", "book_flight"),
        ("Aaj ka mausam kaisa hai", "get_weather"),
        ("Mere account ka balance kitna hai check karo", "check_balance"),
        ("Pizza order kardo Swiggy se", "order_food"),
        ("Kal subah 6 baje ka alarm set karo", "set_alarm"),
        ("Flight booking cancel kardo refund chahiye", "cancel_booking"),
        ("Arijit Singh ke gaane bajao Spotify pe", "play_music"),
        ("Airport pahunchne ka rasta batao", "get_directions"),
        ("Customer care support se baat karwao", "customer_care"),

        # Expanded Real-world Intents
        ("Uber cab book kardo jaldi", "cab_booking"),
        ("Goa me ek accha hotel room book karo", "hotel_booking"),
        ("500 rupaye transfer kardo GPay se", "send_money"),
        ("Mobile recharge kardo 299 wala pack", "recharge_bill"),
        ("Koi acchi thriller movie recommend karo", "movie_recommendation"),
        ("IPL score update batao abhi", "sports_update"),
        ("Meeting ke liye reminder laga do", "set_reminder"),
        ("Bukhar aur sardi ho gayi hai advice do", "health_advice"),
        ("Dawa lene ka time remind karo", "medicine_reminder"),
        ("Amazon pe shoes shopping karni hai", "shopping"),
        ("Aaj ki breaking news batao", "news_update"),
        ("Google pe search karo Python tutorial", "search_web"),
        ("Ek mast funny joke sunao", "tell_joke"),
        ("India ka Prime Minister kaun hai", "general_knowledge"),
        ("Hello kaise ho aap", "greeting"),
        ("Alvida phir milenge bye", "farewell"),
        ("Thank you so much aapki help ke liye", "thank_you"),
        ("Sorry galti se mistake ho gayi", "apology"),
        ("Haan bilkul theek hai", "affirmation"),
        ("Nahi bilkul nahi karna", "negation"),

        # Negative Controls & Out-of-Scope (MUST NOT trigger false positive greetings)
        ("Sanchit mera dost hai", "no_intent_detected"),
        ("Rohit aur Mohit TV dekh rahe hain", "no_intent_detected"),
        ("asdfghjkl random noise 12345", "no_intent_detected"),
    ]

    print(f"\nRunning {len(test_cases)} Intent Verification Tests...\n" + "=" * 60)
    passed = 0
    failed = 0

    for text, expected in test_cases:
        res = p.predict(text, task="intent")
        pred_intent = res["intent"]
        conf = res["confidence"]
        engine = res.get("model_type", "unknown")

        match = (pred_intent == expected)
        if match:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"[{status}] '{text}'")
        print(f"       Expected: {expected} | Got: {pred_intent} ({conf:.1%}) [{engine}]")

    print("=" * 60)
    print(f"Summary: {passed}/{len(test_cases)} passed ({passed/len(test_cases):.1%})")
    assert failed == 0, f"{failed} test(s) failed!"
    print("\nALL INTENT TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
