# Smart-Expense-Tracker

Creating a full-fledged project like a Smart Expense Tracker with machine learning capabilities involves several components, including data collection, model training, and a user interface. Below is a simple version of such a program implemented in Python. This program includes functionality for categorizing and analyzing spending patterns and uses a basic machine learning model for categorization. We will use the `pandas` library for data manipulation, `scikit-learn` for machine learning, and `pickle` to save and load models. Additionally, we'll demonstrate error handling but keep the UI part to a console-based interaction for simplicity.

Before running this code, ensure you have the necessary libraries installed:
```bash
pip install pandas scikit-learn pickle-mixin
```

Here's a simple implementation:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Sample data - Feel free to extend this with your data set
expense_data = {
    'Description': [
        'Grocery shopping at Walmart',
        'Dinner at Italian restaurant',
        'Monthly gym membership',
        'Gas refill',
        'Netflix subscription',
        'Bought a new laptop',
        'Bus ticket',
        'Coffee at Starbucks'
    ],
    'Category': [
        'Groceries',
        'Dining',
        'Fitness',
        'Transportation',
        'Subscription',
        'Electronics',
        'Transportation',
        'Dining'
    ]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(expense_data)

def train_model(df):
    # Split the data
    try:
        X = df['Description']
        y = df['Category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a pipeline that vectorizes the text data then fits a Naive Bayes classifier
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Save the trained model
        with open('expense_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    except Exception as e:
        print("An error occurred during model training:", e)
        raise

def load_model():
    # Load the trained model
    try:
        with open('expense_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        raise
    except Exception as e:
        print("An error occurred while loading the model:", e)
        raise

def categorize_expense(description, model):
    try:
        category = model.predict([description])[0]
        return category
    except Exception as e:
        print("An error occurred during expense categorization:", e)
        raise

def main():
    # Train the model with the data
    train_model(df)

    # Load the model
    model = load_model()

    # Example user input - in a real app this would come from user input
    new_expenses = [
        'Pizza at Domino\'s',
        'Uber ride downtown',
        'Yoga class monthly fee',
        'HBO Max subscription'
    ]

    for expense in new_expenses:
        try:
            category = categorize_expense(expense, model)
            print(f"Expense: {expense} -> Predicted Category: {category}")
        except Exception as e:
            print(f"Could not categorize expense '{expense}':", e)

if __name__ == "__main__":
    main()
```

### Key Components:

1. **Data Preparation**: The sample data includes several expenses with descriptive text and their corresponding categories.
   
2. **Model Training**: A simple Naive Bayes classifier within a pipeline that transforms the text data using TF-IDF vectorization.

3. **Model Persistence**: The model is saved using `pickle` for reuse without retraining.

4. **Expense Categorization**: New expenses are classified based on their descriptions.

5. **Error Handling**: The program includes try-except blocks around key operations to handle errors gracefully.