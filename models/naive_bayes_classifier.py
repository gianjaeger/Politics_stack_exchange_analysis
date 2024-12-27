from sklearn.metrics import classification_report, accuracy_score

def train_and_predict_naive_bayes(all_processed_posts):

    # Filter labeled and unlabeled data
    labeled_posts = all_processed_posts.dropna(subset=['Category'])  # Posts with a category
    unlabeled_posts = all_processed_posts[all_processed_posts['Category'].isna()]  # Posts without a category

    # Split labeled data into training (80%) and testing (20%) sets
    train_data, test_data = train_test_split(labeled_posts, test_size=0.20, random_state=32)

    # Vectorize the Cleaned_Body text
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['Cleaned_Body'])  # Fit on training data
    X_test = vectorizer.transform(test_data['Cleaned_Body'])  # Transform test data
    X_unlabeled = vectorizer.transform(unlabeled_posts['Cleaned_Body'].fillna(""))  # Transform unlabeled data

    # Labels
    y_train = train_data['Category']
    y_test = test_data['Category']

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred_test = classifier.predict(X_test)

    # Calculate and print evaluation metrics
    print("Classifier Performance on Test Data:")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

    # Predict categories for unlabeled posts
    unlabeled_posts['Predicted_Category'] = classifier.predict(X_unlabeled)

    # Return all necessary variables
    return {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "test_data": test_data,
        "unlabeled_posts": unlabeled_posts,
    }
