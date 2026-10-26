"""
Dataset preparation script for 32 intent classes in Hindi-English Code-Mixed (Hinglish),
Pure Hindi (transliterated), and English utterances.

Produces:
    data/processed/intent/train.jsonl
    data/processed/intent/val.jsonl
    data/processed/intent/test.jsonl
"""

import json
import random
from pathlib import Path

random.seed(42)

CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Pune", "Hyderabad", "Jaipur", "Lucknow", "Goa", "Chandigarh", "Ahmedabad"]
DESTINATIONS = ["Airport", "Railway Station", "Connaught Place", "Bandra", "Whitefield", "Cyber City", "Mall", "Hospital", "Metro Station"]
FOODS = ["pizza", "biryani", "burger", "samosa", "chai", "noodles", "thali", "dosa", "chole bhature", "ice cream", "pasta", "momos"]
RESTAURANTS = ["Dominos", "Swiggy", "Zomato", "McDonalds", "KFC", "Haldiram", "Burger King", "Behrouz"]
SONGS = ["Arijit Singh", "Sidhu Moose Wala", "AP Dhillon", "KK", "Atif Aslam", "Lata Mangeshkar", "Diljit Dosanjh", "Shreya Ghoshal"]
MOVIES = ["Pathaan", "Jawan", "3 Idiots", "Dangal", "Sholay", "Mirzapur", "Family Man", "Panchayat", "Stree 2"]
SPORTS = ["cricket", "IPL", "India vs Pakistan", "Virat Kohli", "Rohit Sharma", "football", "World Cup", "CSK", "RCB"]
AMOUNTS = ["500", "1000", "2000", "5000", "10000", "250", "100"]

INTENT_TEMPLATES = {
    # 1. Travel & Logistics
    "book_flight": [
        "mujhe {origin} se {dest} ki flight ticket book karni hai",
        "flight tickets from {origin} to {dest} book karo",
        "kal subah {origin} se {dest} ke liye flight chahiye",
        "{origin} to {dest} flight prices aur availability check karo",
        "i want to book a flight ticket to {dest} from {origin}",
        "kripya {origin} se {dest} ke liye sasti flight dhoondo",
        "can you reserve a seat on the next flight to {dest}",
        "flight booking kardo {dest} ke liye please",
        "{origin} se {dest} morning flight kitne baje hai",
        "direct flight available hai kya {origin} se {dest} tak",
        "air ticket booking kar do {dest} ke liye kal subah",
        "cheapest flight to {dest} book kar do",
    ],
    "cab_booking": [
        "ek cab book kardo {dest} jane ke liye",
        "uber cab bulao jaldi se",
        "ola ride book kardo office ke liye",
        "mujhe {dest} tak taxi chahiye",
        "book a cab from {origin} to {dest}",
        "nearest auto ya cab dhundo",
        "cab kitne der me aayegi check karo",
        "sedan cab book kar do 4 logon ke liye",
        "uber go book kardo urgent hai",
        "rasta lamba hai ek taxi book kar do",
        "airport ke liye cab schedule kardo kal subah",
    ],
    "hotel_booking": [
        "{city} me ek accha hotel room book karo",
        "hotel reservation karni hai 2 nights ke liye {city} me",
        "can you book a 3-star hotel near {dest}",
        "{city} me sasta aur accha hotel room chahiye",
        "oyo room book kar do couple ke liye",
        "hotel check-in timings kya hai {city} me",
        "luxury resort book kardo {city} weekend ke liye",
        "hotel room booking confirmation chahiye",
        "best hotel deals in {city} dikhao aur book karo",
        "family suite book kardo {city} hotel me",
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
        "rasta dikhao airport ka jaldi",
        "navigation start karo {dest} ke liye",
        "direction batao sabse chhota rasta kaunsa hai",
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
        "flight ticket cancel karke refund account me daalo",
    ],

    # 2. Daily Utilities & Weather
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
        "kya kal baarish hone wali hai {city} me",
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
        "SBI account balance check karo",
    ],
    "send_money": [
        "{amount} rupaye transfer kardo friend ko",
        "send money via UPI to Rahul",
        "GPay se {amount} bhejo Papa ko",
        "PhonePe par payment transfer kardo",
        "account me {amount} transfer karo jaldi",
        "send {amount} rupees to my brother",
        "UPI payment karni hai QR code scan karke",
        "paise bhejo bank account me {amount}",
        "transfer money to savings account",
        "Paytm se payment send kardo",
    ],
    "recharge_bill": [
        "mobile recharge kardo 299 wala pack",
        "bijli ka bill bharna hai electricity bill pay karo",
        "Jio number recharge kar do 1 month ke liye",
        "Airtel broadband bill payment karo",
        "DTH recharge kardo Tata Play",
        "water bill pay karna hai online",
        "metro card recharge kardo 500 rupaye se",
        "fastag recharge kardo 1000 rupees",
        "bill payment reminder set karo",
        "phone recharge kar do urgently validity khatam ho gayi",
    ],

    # 3. Lifestyle, Food & Entertainment
    "order_food": [
        "ek {food} order karo {restaurant} se",
        "lunch me {food} order kardo {restaurant} se",
        "can you order some {food} for dinner",
        "kuch khana mangwa do bhook lag rahi hai",
        "order food from the nearest restaurant",
        "{restaurant} se {food} deliver karwa do",
        "ek cup chai aur {food} order kardo",
        "dinner order kar do 2 logo ke liye",
        "{restaurant} se {food} manga do please",
        "food delivery order place karo jaldi",
        "swiggy se pizza aur burger mangwa do",
    ],
    "play_music": [
        "koi acha sa Hindi gaana bajao",
        "play {song} top romantic songs",
        "music chalao Spotify pe party songs",
        "play some relaxing instrumental music",
        "{song} ke trending gaane bajao",
        "volume badhao aur Bollywood hits chalao",
        "play my favorite playlist please",
        "gaana bajao koi sukoon wala",
        "punjabi party songs play kardo",
        "youtube music pe trending songs sunao",
        "chill vibes playlist play karo",
    ],
    "movie_recommendation": [
        "koi acchi Bollywood movie recommend karo weekend ke liye",
        "Netflix pe best suspense thriller movie batao",
        "aaj raat kaunsi film dekhu suggest something good",
        "IMDb top rated movies ki list dikhao",
        "comedy web series suggest karo koi funny",
        "koi nayi action movie release hui hai kya",
        "romantic movie recommendation chahiye",
        "{movie} jaisi koi aur movie batao",
        "weekend binge watch ke liye series suggest karo",
        "Prime Video par trending movies kaunsi hain",
    ],
    "sports_update": [
        "aaj ka {sport} match ka score kya hai",
        "IPL live score update batao abhi",
        "India vs Pakistan match kaun jeeta",
        "Virat Kohli ne kitne run banaye aaj",
        "points table dikhao IPL ki",
        "World Cup match schedule kya hai kal ka",
        "who won today's football match",
        "cricket commentary aur scorecard dikhao",
        "aaj kis kis team ka match hai",
        "next match kab hai Indian cricket team ka",
    ],

    # 4. Alarms, Tasks & Health
    "set_alarm": [
        "kal subah 6 baje ka alarm set karo",
        "set an alarm for 7:30 AM tomorrow",
        "subah jaldi uthne ke liye 5 baje alarm laga do",
        "ek ghante baad ka timer set karo",
        "alarm laga do subah 6:00 baje ka",
        "wake me up tomorrow at 6:30 in the morning",
        "kal ka alarm 7 baje set kar do",
        "subah 5:30 ka alarm turn on karo",
        "15 minute ka timer lagao cooking ke liye",
        "set alarm for morning meeting at 8 AM",
    ],
    "set_reminder": [
        "remind me at 4 PM to attend the meeting",
        "set a reminder for medicine at 9 PM",
        "reminder laga do meeting ke liye kal dopahar 2 baje",
        "mujhe yaad dilana shaam ko milk lana hai",
        "set a reminder to call mom at 8 PM",
        "kal project submit karna hai reminder set kardo",
        "doctor appointment ka reminder laga do kal 11 baje",
        "don't forget reminder set karo car service ke liye",
        "birthday reminder save karo 15th October",
        "gym jane ka reminder set kardo shaam 6 baje",
    ],
    "health_advice": [
        "mujhe kal se bukhar aur sardi hai kya karu",
        "sir me bohot tez dard ho raha hai koi dawa batao",
        "pet dard ke liye kya gharelu upaay hain",
        "fever and headache symptoms ke liye doctor consult kaise kare",
        "throat pain aur cough ho raha hai advice do",
        "blood pressure high ho gaya hai kya kare",
        "healthy diet plan suggest karo weight loss ke liye",
        "immunity boost karne ke tips batao",
        "back pain relief ke exercises kya hain",
        "symptoms of viral infection kya hote hain",
    ],
    "medicine_reminder": [
        "dawa lene ka time remind karo raat ko",
        "BP ki goli lene ka reminder laga do subah 8 baje",
        "tablet lene ka schedule set karo lunch ke baad",
        "medicine dosage time notify karo daily",
        "antibiotics lene ka reminder 3 baje set karo",
        "dawa yaad dilana khane se pehle",
        "paracetamol lene ka time ho gaya kya",
        "night dose medicine reminder on kardo",
    ],
    "shopping": [
        "Amazon pe best wireless earphones dhoondo",
        "Flipkart par sports shoes ki sale check karo",
        "ek naya mobile phone khareedna hai under 20000",
        "smart watch buy karne ke deals dikhao",
        "online shopping karna hai discount coupons batao",
        "laptop prices compare karo Amazon aur Flipkart pe",
        "running shoes order karne hain size 9",
        "men's t-shirts online purchase karni hain",
        "price drop alert set karo is product pe",
        "shopping cart me add kardo ye product",
    ],

    # 5. Information, Search & Conversation
    "general_knowledge": [
        "India ka Prime Minister kaun hai",
        "who is the CEO of Google",
        "Taj Mahal kisne banwaya tha aur kahan hai",
        "photosynthesis kya hota hai simple words me samjhao",
        "what is artificial intelligence and machine learning",
        "solar system me kitne planets hote hain",
        "gravity ki khoj kisne ki thi",
        "Mount Everest ki height kitni hai",
        "explain quantum computing in Hindi",
        "Bharat ka samvidhan kab lagu hua tha",
    ],
    "news_update": [
        "aaj ki breaking news batao Hindi me",
        "latest headlines kya hain National aur International",
        "stock market aur business news update do",
        "aaj ke top samachar dikhao",
        "technology world me kya naya chal raha hai news",
        "weather and election news update batao",
        "cricket news aur sports headlines dikhao",
        "today's top news stories summarize karo",
    ],
    "search_web": [
        "Google par search karo best Python tutorials",
        "internet pe find out karo how to make paneer butter masala",
        "web search karo upcoming tech events in India",
        "search this on Google and give me summary",
        "online dhoondo best libraries for NLP in Python",
        "wiki page dhundo APJ Abdul Kalam ke baare me",
        "lookup facts about Indian space program ISRO",
    ],
    "greeting": [
        "Hello kaise ho aap",
        "Hi good morning",
        "Namaste ji kya haal chaal",
        "Hey what's up",
        "Hello bot how are you today",
        "Namaskar kaisa chal raha hai sab",
        "Good evening bhai kaise ho",
        "Hey there nice to meet you",
        "Salam kya haal hai",
        "Sat sri akal ji",
        "Hi bot kaisa hai tu",
        "Good afternoon hope you are doing well",
    ],
    "farewell": [
        "Alvida phir milenge",
        "Bye bye have a nice day",
        "Good night shubh ratri",
        "See you later take care",
        "Chalta hoon ab kal baat karenge",
        "Bye take care of yourself",
        "Signing off goodbye",
        "Good night sweet dreams",
        "Phir milte hain bye",
        "Catch you later my friend",
    ],
    "thank_you": [
        "Thank you so much aapki help ke liye",
        "Bahut bahut shukriya aapka",
        "Thanks a lot you are very helpful",
        "Dhanyawad bhai maza aa gaya",
        "Really appreciate your help thank you",
        "Shukriya meri madad karne ke liye",
        "Thanks buddy great job",
        "Bohot badiya thank you very much",
    ],
    "apology": [
        "Sorry galti se type ho gaya",
        "Mujhe maaf karna meri mistake thi",
        "I am so sorry for the delay",
        "Maafi chahta hoon galat information di",
        "Apologies my bad",
        "Sorry bhai dhyan nahi diya tha",
        "Extremely sorry about that",
    ],
    "tell_joke": [
        "koi funny joke sunao hasne ka man hai",
        "ek mast chutkula sunao Hindi me",
        "tell me a funny programmer joke",
        "kuch funny batao bore ho raha hoon",
        "hasao mujhe koi mast mazedaar joke se",
        "make me laugh with a funny story",
        "Santa Banta ka joke sunao please",
        "koi comedy dialogue ya joke sunao",
    ],
    "affirmation": [
        "haan bilkul sahi kaha aapne",
        "yes exactly that is correct",
        "haanji main agree karta hoon",
        "bilkul sahi baat hai",
        "yes please proceed further",
        "haan confirm kardo isko",
        "sure definitely go ahead",
        "theek hai haan kar do",
    ],
    "negation": [
        "nahi bilkul nahi chahiye mujhe",
        "no that is not what I wanted",
        "nahin cancel kardo mat karo",
        "no don't do that stop",
        "nahi aisa nahi hai galat hai",
        "no I disagree completely",
        "nahi rehne do abhi nahi karna",
        "no thank you not interested",
    ],

    # 6. Customer Care, Emergency & Crisis
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
        "support team ko call lagao helpline number do",
        "service bahut kharab hai customer care bulao",
    ],
    "emergency_help": [
        "bachao mujhe emergency help chahiye",
        "police ko call lagao turant 100 number pe",
        "ambulance bulao road par accident ho gaya hai",
        "fire brigade ko call karo aag lag gayi hai",
        "chor aa gaye ghar me help please police bulao",
        "medical emergency hai ambulance send karo jaldi",
        "save me I am in danger emergency",
        "help someone is attacking please call police",
        "emergency helpline connect karo urgently",
    ],
    "mental_health_distress": [
        "Mujhe marna",
        "mujhe marna hai jeene ka man nahi karta",
        "i want to kill myself",
        "i want to end my life, feeling hopeless",
        "sab khatam ho gaya meri zindagi bekar hai",
        "bahut depressed hoon rona aa raha hai constantly",
        "jeena nahi chahta main ab aur",
        "khud ko hurt karne ka man kar raha hai",
        "nobody cares about me I want to die",
        "zindagi se tang aa chuka hoon give up karna chahta hoon",
        "bohot akela aur depressed feel ho raha hai koi sahara nahi",
        "suicidal thoughts aa rahe hain mujhe please help",
        "thak gaya hoon is zindagi se sab chhod ke marna chahta hoon",
        "i feel so alone and want to end it all",
        "mujhe marna hai koi rasta nahi bacha",
    ],

    "threat_violence": [
        "Sanchit ko marunga",
        "main use peet dunga ache se",
        "jaan se maar dunga tujhe",
        "use maar dalunga main",
        "usko thappad marunga",
        "I will beat him up",
        "I will punch you in the face",
        "use sabak sikhaunga mar mar ke",
        "goli maar dunga tujhe dekh lena",
        "use zinda nahi chhodunga",
        "uspar attack karunga main",
        "I will attack and hurt them",
        "tujhe khatam kar dunga",
        "use peetna hai mujhe",
    ],

    # 7. Explicit Out-of-Domain / No Intent
    "no_intent_detected": [
        "asdfghjkl qwertyuiop",
        "kuch bhi randomly likh raha hoon yahan",
        "the sky is blue and grass is green on the hills",
        "1234567890 987654321",
        "bla bla bla testing 123",
        "xyz abc def ghi jkl mno",
        "hmmm ahhh ok okies",
        "ek tha raja ek thi rani dono mar gaye khatam kahani",
        "just random keystrokes zxcvbnm",
        "nothing in particular here today",
        "hahaha hehehe hohoho lol",
        "random string of meaningless words",
        "table chair window door ceiling floor",
        "typing text without asking anything",
        "foobar baz qux corge grault",
        "random gibberish asdasdasd",
        "Sanchit mera dost hai",
        "Rohit aur Mohit cricket dekh rahe hain",
        "ye ek normal sentence hai bina kisi intent ke",
        "simple casual talk without action",
    ],
}

def generate_intent_dataset():
    train_samples = []
    val_samples = []
    test_samples = []

    samples_per_class = 150  # 32 * 150 = 4,800 total samples

    for intent, templates in INTENT_TEMPLATES.items():
        class_samples = []
        # Cycle through templates filling slots
        while len(class_samples) < samples_per_class:
            for tmpl in templates:
                orig = random.choice(CITIES)
                dest = random.choice([c for c in CITIES if c != orig] + DESTINATIONS)
                city = random.choice(CITIES)
                food = random.choice(FOODS)
                rest = random.choice(RESTAURANTS)
                song = random.choice(SONGS)
                movie = random.choice(MOVIES)
                sport = random.choice(SPORTS)
                amt = random.choice(AMOUNTS)

                text = tmpl.format(
                    origin=orig, dest=dest, city=city, food=food,
                    restaurant=rest, song=song, movie=movie,
                    sport=sport, amount=amt,
                )

                # Add some casing/punctuation jitter
                if random.random() < 0.2:
                    text = text.lower()
                elif random.random() < 0.2:
                    text = text.strip(".!?") + "?"
                elif random.random() < 0.2:
                    text = text.strip(".!?") + "!"

                class_samples.append({"text": text, "intent": intent})
                if len(class_samples) >= samples_per_class:
                    break

        random.shuffle(class_samples)
        train_end = int(0.80 * samples_per_class)
        val_end = int(0.92 * samples_per_class)

        train_samples.extend(class_samples[:train_end])
        val_samples.extend(class_samples[train_end:val_end])
        test_samples.extend(class_samples[val_end:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    out_dir = Path("data/processed/intent")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(out_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(INTENT_TEMPLATES)} classes:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    print("Classes:", sorted(list(INTENT_TEMPLATES.keys())))

if __name__ == "__main__":
    generate_intent_dataset()
