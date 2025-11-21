import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def create_dataset():
    """Return texts and labels for a small spam classification task."""
    data = [
        # spam
        ("spam", "Winner! Claim your free laptop now, limited time offer."),
        ("spam", "Your bank account will be closed, confirm your details immediately."),
        ("spam", "Exclusive deal on crypto tokens, huge returns guaranteed."),
        ("spam", "Get cheap software licenses, 90 percent discount for today only."),
        ("spam", "Important: your password has expired, click here to reset."),
        ("spam", "Earn passive income by joining our marketing program."),
        ("spam", "Final reminder: unclaimed reward waiting, verify your address."),
        ("spam", "Free access to streaming movies, register with your card."),
        ("spam", "Lottery winner notice, respond with your full name and country."),
        ("spam", "Special pharmacy promotion, buy one get one free."),

        # ham
        ("ham", "Can you send me the latest firewall configuration file?"),
        ("ham", "Here are the minutes from yesterday's SOC meeting."),
        ("ham", "Please check the backup logs for any failed jobs."),
        ("ham", "Let's schedule a call to discuss the new VPN rollout."),
        ("ham", "I pushed a fix for the login timeout bug to the repository."),
        ("ham", "Reminder: security awareness training starts at 3 PM."),
        ("ham", "Coffee break at 11 in the kitchen?"),
        ("ham", "The monitoring dashboard alert was a false positive."),
        ("ham", "Your access to the lab has been approved, badge is ready."),
        ("ham", "I updated the wiki page for the incident response plan."),
    ]

    labels, texts = zip(*data)
    return list(texts), list(labels)


def encode_labels(labels):
    """Map 'ham' -> 0 and 'spam' -> 1."""
    mapping = {"ham": 0, "spam": 1}
    return np.array([mapping[label] for label in labels])


def train_model():
    texts, labels_str = create_dataset()
    y = encode_labels(labels_str)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.3, random_state=0, stratify=y
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Inspect predictions on a few custom messages
    samples = [
        "Your mailbox storage is full, log in to avoid deactivation.",
        "Thanks for sending the incident report.",
        "Urgent: verify your payment information to continue using the service.",
    ]
    sample_features = vectorizer.transform(samples)
    sample_preds = model.predict(sample_features)

    print("\nExample predictions:")
    for msg, pred in zip(samples, sample_preds):
        label = "spam" if pred == 1 else "ham"
        print(f"- [{label}] {msg}")

    # Visualisation: histogram of message lengths (spam vs ham)
    lengths = np.array([len(t.split()) for t in texts])
    labels_int = y

    spam_lengths = lengths[labels_int == 1]
    ham_lengths = lengths[labels_int == 0]

    plt.figure()
    plt.hist(ham_lengths, bins=range(0, 30, 2), alpha=0.7, label="ham")
    plt.hist(spam_lengths, bins=range(0, 30, 2), alpha=0.7, label="spam")
    plt.xlabel("Message length (words)")
    plt.ylabel("Count")
    plt.title("Length distribution for spam vs ham")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task_3/length_histogram.png", dpi=200)
    print("Saved length_histogram.png")


if __name__ == "__main__":
    train_model()
