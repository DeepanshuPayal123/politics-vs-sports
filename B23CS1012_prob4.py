"""
Assignment 1 - Problem 4: Sports vs Politics Classifier
Name: [Your Name Here]
Roll Number: [Your Roll Number Here]
Date: January 19, 2026

Description:
This script implements a text classifier to distinguish between 'Sports' and 'Politics'
articles. It compares three machine learning algorithms:
1. Naive Bayes (MultinomialNB)
2. Support Vector Machine (SVM)
3. Logistic Regression

Features:
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
- Includes a built-in dataset for immediate execution (no external file needed).
- Prints accuracy and detailed classification reports for model comparison.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_data():
    """
    Returns a labeled dataset of Sports and Politics headlines.
    Dataset size has been increased to 100 samples to ensure >80% accuracy.
    """
    
    # Sports Data (50 distinct examples)
    sports_texts = [
        "The match ended in a thrilling draw after overtime.",
        "Cricket world cup finals will be held next month.",
        "The football manager was sacked after a series of losses.",
        "She won the tennis grand slam title in straight sets.",
        "The team scored a hat-trick in the final minutes.",
        "Olympic gold medalist breaks another world record.",
        "The defensive strategy failed against the strong attack.",
        "The wicket keeper dropped a crucial catch.",
        "Basketball playoffs are starting this weekend.",
        "The striker is suffering from a hamstring injury.",
        "F1 driver takes pole position for the Grand Prix.",
        "The referee made a controversial decision during the game.",
        "Athletes are training hard for the upcoming season.",
        "The stadium was packed with cheering fans.",
        "He is the highest run-scorer in the tournament history.",
        "The boxing match lasted for twelve rounds.",
        "Swimming championship results were announced today.",
        "The hockey team qualified for the semi-finals.",
        "Badminton world rankings have been updated.",
        "The coach praised the team's spirit and effort.",
        "A new transfer record was set in the premier league.",
        "The marathon winner finished in under two hours.",
        "Rugby rules are often confusing for new spectators.",
        "The gymnastics routine scored a perfect ten.",
        "Volleyball finals will be broadcast live tonight.",
        "The quarterback threw a touchdown pass in the last second.",
        "NBA finals MVP was awarded to the star player.",
        "FIFA announced the host country for the next World Cup.",
        "The bowler took five wickets in a single innings.",
        "Serena Williams announces retirement from professional tennis.",
        "The grandmaster checkmated his opponent in 20 moves.",
        "Formula 1 race was delayed due to heavy rain on the track.",
        "The underdog team defeated the defending champions.",
        "UFC fighter retained the championship belt via knockout.",
        "The cycling tour covers over 3000 kilometers of terrain.",
        "Golf tournament leader is two strokes ahead of the pack.",
        "The goalkeeper made a spectacular save to deny the goal.",
        "Table tennis championship finals are scheduled for Sunday.",
        "The wrestler won gold in the freestyle category.",
        "Baseball season opener attracted a massive crowd.",
        "The sprinter was disqualified for a false start.",
        "Chess is considered a sport of the mind.",
        "The skiing competition was cancelled due to avalanche risk.",
        "The rowing team synchronized perfectly to win the regatta.",
        "Surfing championship held at the famous coastline.",
        "The archer hit the bullseye to win the gold medal.",
        "Skateboarding made its debut at the summer games.",
        "The karate kid won the local martial arts tournament.",
        "Paralympic athletes inspire millions with their dedication.",
        "The derby horse race is the most exciting two minutes in sports."
    ]
    
    # Politics Data (50 distinct examples)
    politics_texts = [
        "The election results are delayed due to recounting.",
        "Parliament passed the new bill after a long debate.",
        "The president gave a speech on national security.",
        "Tax reform policies are being discussed in the senate.",
        "Vote for the candidate who promises real change.",
        "The minister resigned today amidst the scandal.",
        "Diplomatic talks with foreign leaders were successful.",
        "Senate voting starts tomorrow morning.",
        "The government announced a new budget plan.",
        "Protests erupted over the new legislation.",
        "The opposition party criticized the prime minister.",
        "Democracy requires active participation from citizens.",
        "The campaign trail visits three states this week.",
        "Policy changes will affect healthcare funding.",
        "The governor vetoed the proposed law.",
        "International trade agreements were signed yesterday.",
        "The mayor is running for re-election next year.",
        "Political analysts predict a close race.",
        "The supreme court ruled on the constitutional matter.",
        "Cabinet members met to discuss the economic crisis.",
        "Voting booths opened early for the general election.",
        "The senator filibustered for ten hours.",
        "Lobbyists are pushing for stricter environmental laws.",
        "The referendum outcome was unexpected.",
        "Foreign policy is a key topic in the debate.",
        "The dictator was overthrown by a military coup.",
        "Human rights organizations criticized the new decree.",
        "The summit of world leaders addressed climate change policy.",
        "Congress is divided on the issue of immigration reform.",
        "The ballot boxes were secured by federal agents.",
        "A coalition government was formed after the hung parliament.",
        "The chancellor announced tax cuts for small businesses.",
        "Voter turnout was historically low in this district.",
        "The impeachment trial began in the upper house.",
        "Socialist and conservative parties clashed in the debate.",
        "The diplomat was recalled to the capital for consultations.",
        "Legislative assembly approved the new education bill.",
        "The head of state visited the disaster-stricken area.",
        "Campaign finance laws are being reviewed by the committee.",
        "The polling station closed at 8 PM sharp.",
        "Exit polls suggest a landslide victory for the incumbent.",
        "The prime minister dissolved the parliament for early elections.",
        "Constitutional amendments require a two-thirds majority.",
        "The geopolitical tension in the region is escalating.",
        "Local council elections will be held next Tuesday.",
        "The manifesto promises free healthcare for all citizens.",
        "Bilateral talks focused on border security and trade.",
        "The senator introduced a bill to protect whistleblowers.",
        "Political corruption charges were filed against the official.",
        "The administration is facing pressure to reduce inflation."
    ]

    # Combine data and create labels (0 for Sports, 1 for Politics)
    texts = sports_texts + politics_texts
    labels = ["Sports"] * len(sports_texts) + ["Politics"] * len(politics_texts)
    
    return texts, labels

def main():
    print("--- Loading Data ---")
    texts, labels = get_data()
    print(f"Total samples: {len(texts)}")
    print(f"Sports samples: {labels.count('Sports')}")
    print(f"Politics samples: {labels.count('Politics')}")
    print("-" * 30)

    # 1. Feature Extraction (TF-IDF)
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(texts)
    y = labels

    # 2. Train-Test Split (80% Train, 20% Test for better learning)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary of models to compare
    models = {
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression()
    }

    # 3. Model Training and Evaluation
    print("\n--- Model Comparison ---")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test data
        predictions = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, predictions)
        
        print(f"Results for {name}:")
        print(f"Accuracy: {acc:.2f} ({acc*100:.0f}%)")
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("-" * 30)

if __name__ == "__main__":
    main()
