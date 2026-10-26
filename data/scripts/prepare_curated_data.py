"""
Script to generate balanced, high-quality Hinglish training data for:
1. Intent Detection (multi-domain: flight, weather, banking, food, alarm, cancel, music, directions, greeting)
2. Sentiment Analysis (balanced 12k positive, negative, neutral samples)
"""

import json
import random
from pathlib import Path

random.seed(42)

# ==============================================================================
# 1. INTENT DATASET
# ==============================================================================
intent_templates = {
    "book_flight": [
        "mujhe {origin} se {dest} ki flight ticket book karni hai",
        "flight tickets from {origin} to {dest} book karo",
        "kal subah {origin} se {dest} ke liye flight chahiye",
        "{origin} to {dest} flight prices aur availability check karo",
        "i want to book a ticket to {dest} from {origin}",
        "kripya {origin} se {dest} ke liye sasti flight dhoondo",
        "can you reserve a seat on the next flight to {dest}",
        "flight booking kardo {dest} ke liye please",
        "{origin} se {dest} morning flight kitne baje hai",
        "direct flight available hai kya {origin} se {dest} tak",
    ],
    "get_weather": [
        "aaj {city} me mausam kaisa hai",
        "{city} me aaj barish hogi kya",
        "what is the weather forecast for {city} today",
        "aaj ka temperature kitna hai {city} me",
        "{city} ka weather report batao",
        "kal {city} me sunny hoga ya cloudy",
        "is it going to rain in {city} this evening",
        "{city} me garmi kitni hai aaj",
        "mausam ka haal batao {city} ke liye",
        "humidity level kya hai {city} me",
    ],
    "check_balance": [
        "mere account ka balance kitna hai check karo",
        "bank balance batao mere savings account ka",
        "show me my current account balance please",
        "khate me kitne paise bache hain",
        "check my account balance and recent transactions",
        "mera bank statement dikhao last month ka",
        "account me kitna balance hai abhi",
        "current balance kitna hai mere wallet me",
        "can you check my available balance",
        "mere account me kitne rupaye hain",
    ],
    "order_food": [
        "ek pizza order karo Dominos se",
        "lunch me biryani order kardo Swiggy se",
        "can you order some burgers and fries for dinner",
        "kuch khana mangwa do bhook lag rahi hai",
        "order food from the nearest Chinese restaurant",
        "Zomato se pizza deliver karwa do",
        "ek cup chai aur samosa order kardo",
        "dinner order kar do 2 logo ke liye",
        "swiggy se ice cream manga do please",
        "food delivery order place karo jaldi",
    ],
    "set_alarm": [
        "kal subah 6 baje ka alarm set karo",
        "set an alarm for 7:30 AM tomorrow",
        "subah jaldi uthne ke liye 5 baje alarm laga do",
        "remind me at 4 PM to attend the meeting",
        "ek ghante baad ka timer set karo",
        "alarm laga do subah 6:00 baje ka",
        "set a reminder for medicine at 9 PM",
        "wake me up tomorrow at 6:30 in the morning",
        "kal ka alarm 7 baje set kar do",
        "reminder laga do meeting ke liye kal dopahar 2 baje",
    ],
    "cancel_booking": [
        "meri flight booking cancel kardo please",
        "cancel my hotel reservation immediately",
        "ticket cancel karni hai refund kitna milega",
        "please cancel the order I placed earlier",
        "booking cancel kar do aur refund process karo",
        "i want to cancel my cab booking right now",
        "cancel this reservation and send confirmation",
        "train ticket cancel kaise kare refund chahiye",
        "booking cancellation request initiate karo",
        "mera order cancel kardo mujhe nahi chahiye",
    ],
    "play_music": [
        "koi acha sa Hindi gaana bajao",
        "play Arijit Singh top romantic songs",
        "music chalao Spotify pe party songs",
        "play some relaxing instrumental music",
        "sidhu moose wala ke gaane bajao",
        "volume badhao aur Bollywood hits chalao",
        "play my favorite playlist please",
        "gaana bajao koi sukoon wala",
        "punjabi party songs play kardo",
        "youtube music pe trending songs sunao",
    ],
    "get_directions": [
        "yahan se {dest} ka rasta batao",
        "how do I get to {dest} from here",
        "Google Maps pe {dest} ka route dikhao",
        "{dest} pahunchne me kitna time lagega",
        "show me the fastest route to {dest}",
        "nearest metro station ka direction batao",
        "{dest} ke liye best route kaunsa hai",
        "traffic kaisa hai raste me {dest} tak",
        "rasta dikhao airport ka",
        "navigation start karo {dest} ke liye",
    ],
    "customer_care": [
        "customer support se baat karwao",
        "i want to speak with a customer care executive",
        "mera issue resolve nahi hua support team se connect karo",
        "help desk ka number chahiye mujhe",
        "complaint register karni hai service ke khilaf",
        "customer service agent se connect kardo please",
        "technical support help chahiye urgently",
        "executive se baat karni hai complaint hai",
        "talk to an agent regarding my issue",
        "support team ko call lagao",
    ]
}

cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Pune", "Hyderabad", "Jaipur", "Lucknow", "Goa"]
destinations = ["Airport", "Railway Station", "Connaught Place", "Bandra", "Whitefield", "Cyber City", "Mall", "Hospital"]

intent_train_samples = []
intent_val_samples = []

for intent, templates in intent_templates.items():
    samples_for_intent = []
    for _ in range(50):
        for tmpl in templates:
            orig = random.choice(cities)
            dest = random.choice([c for c in cities if c != orig] + destinations)
            city = random.choice(cities)
            text = tmpl.format(origin=orig, dest=dest, city=city)
            samples_for_intent.append({"text": text, "intent": intent})
    
    random.shuffle(samples_for_intent)
    split_pt = int(len(samples_for_intent) * 0.85)
    intent_train_samples.extend(samples_for_intent[:split_pt])
    intent_val_samples.extend(samples_for_intent[split_pt:])

random.shuffle(intent_train_samples)
random.shuffle(intent_val_samples)

# Save intent dataset
intent_out_dir = Path("data/processed/intent")
intent_out_dir.mkdir(parents=True, exist_ok=True)

with open(intent_out_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for s in intent_train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(intent_out_dir / "val.jsonl", "w", encoding="utf-8") as f:
    for s in intent_val_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(intent_out_dir / "test.jsonl", "w", encoding="utf-8") as f:
    for s in intent_val_samples[:300]:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Generated Intent: {len(intent_train_samples)} train, {len(intent_val_samples)} val samples across {len(intent_templates)} classes.")

# ==============================================================================
# 2. SENTIMENT DATASET (Balanced high-signal subset)
# ==============================================================================
sentiment_raw_train = Path("data/processed/sentiment/train.jsonl")
by_label = {0: [], 1: [], 2: []}

if sentiment_raw_train.exists():
    with open(sentiment_raw_train, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lbl = int(item["label"])
            if lbl in by_label:
                by_label[lbl].append(item)

# Balance to 4,000 per class (12,000 total)
target_per_class = min(4000, min(len(v) for v in by_label.values()))

# Inject high-signal Hinglish seed examples for robustness
curated_sentiment_seeds = [
    # Positive (2)
    ("Yeh movie bahut achhi thi, casting aur direction mind blowing tha!", 2),
    ("Product ki quality bahut badiya hai, highly recommended!", 2),
    ("Khaana bohot swadisht tha, maza aa gaya!", 2),
    ("Staff was super helpful and friendly, wonderful experience!", 2),
    ("Best service ever! Bilkul perfect timing aur smooth delivery.", 2),
    ("Zabardast phone hai, camera quality ekdum lajawab!", 2),
    ("Mera experience bohot badhiya raha, love it!", 2),
    ("Paisa vasool movie, pure entertainment!", 2),
    ("Outstanding performance by the actors, truly amazing!", 2),
    ("Bohot accha product hai, definitely buy this!", 2),
    
    # Negative (0)
    ("Khaana ekdum bekar tha, waste of money and time.", 0),
    ("Worst customer care ever, phone bhi attend nahi karte.", 0),
    ("Bohot hi ghatiya aur bakwas service hai inki.", 0),
    ("Product completely damaged mila, bilkul fraud seller hai.", 0),
    ("Third class experience, don't ever buy from them.", 0),
    ("Film bohot boring aur pathetic thi, bilkul mat dekhna.", 0),
    ("Worst app, har second crash ho raha hai.", 0),
    ("Disappointed with the quality, very cheap plastic material.", 0),
    ("Terrible delivery, 2 ghante late aaya khana thanda ho gaya.", 0),
    ("Ekdum bakwas, totally disappointed.", 0),

    # Neutral (1)
    ("Average movie thi, ek baar dekh sakte hain.", 1),
    ("Flight timing change ho gaya, okayish service.", 1),
    ("Normal sa phone hai, kuch khas features nahi hain.", 1),
    ("Khaana theek thaak tha, na bohot accha na kharab.", 1),
    ("Price ke hisab se okay product hai.", 1),
    ("Meeting kal subah 10 baje scheduled hai.", 1),
    ("Package receive ho gaya hai today.", 1),
    ("Standard quality hai, usual daily use ke liye fine.", 1),
    ("Theek hai, kaam chal jayega.", 1),
    ("Average speed hai internet ki yahan.", 1),
]

for text, lbl in curated_sentiment_seeds * 50:
    by_label[lbl].append({"text": text, "label": lbl})

balanced_train = []
balanced_val = []

for lbl in [0, 1, 2]:
    random.shuffle(by_label[lbl])
    samples = by_label[lbl][:target_per_class]
    split_pt = int(len(samples) * 0.85)
    balanced_train.extend(samples[:split_pt])
    balanced_val.extend(samples[split_pt:])

random.shuffle(balanced_train)
random.shuffle(balanced_val)

sent_out_dir = Path("data/processed/sentiment")
with open(sent_out_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for s in balanced_train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(sent_out_dir / "val.jsonl", "w", encoding="utf-8") as f:
    for s in balanced_val:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Generated Sentiment: {len(balanced_train)} train, {len(balanced_val)} val samples (balanced across 3 classes).")
