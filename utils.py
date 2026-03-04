from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def categorize(text):
    t = text.lower()
    if "atm" in t:
        return "ATM Issue"
    elif "card" in t:
        return "Card Issue"
    elif "loan" in t:
        return "Loan Issue"
    elif "login" in t or "app" in t or "internet banking" in t:
        return "App / Login Issue"
    elif "transaction" in t:
        return "Transaction Issue"
    else:
        return "Other"

def severity(text):
    t = text.lower()
    if "deducted" in t or "fraud" in t or "balance reduced" in t:
        return "High"
    elif "not working" in t or "blocked" in t:
        return "Medium"
    else:
        return "Low"

def generate_response(text):
    return (
        "Dear Customer,\n\n"
        "Thank you for bringing this issue to our attention. "
        "We sincerely regret the inconvenience caused. "
        "Our support team is currently reviewing your complaint "
        "and will resolve it as soon as possible. "
        "If the issue involves a financial transaction, "
        "please allow up to 24 hours for verification.\n\n"
        "Regards,\nBank Support Team"
    )