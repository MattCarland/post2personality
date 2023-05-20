from scripts.preprocessing import preprocessing
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB





df2 = pd.read_csv("twitter_MBTI.csv")
df2.drop(df2.columns[0], axis=1, inplace=True)
df2 = df2.rename(columns={'text': 'text', 'label': 'type'})


df3 = pd.read_csv("mbti_1.csv")
df3 = df3[['posts', 'type']]
df3 = df3.rename(columns={'posts': 'text', 'type': 'type'})


#combining two dataframes and outout is ['text'] and ['type']
combined_df = pd.concat([df2, df3], axis=0)

#outputs["clean_text"]
print(preprocessing(combined_df))

#instantiating X and y
X = combined_df['clean_text']
y = combined_df['type']

#Vectorizing X
vectorizer = TfidfVectorizer()
vectorized_documents = vectorizer.fit_transform(X)
vectorized_documents = pd.DataFrame(
    vectorized_documents.toarray(),
    columns = vectorizer.get_feature_names_out()
)

vectorized_documents



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vectorized_documents, y, test_size=0.2, random_state=42)


# Instantiate and fit the classifier on the training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate the accuracy score
accuracy = balanced_accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)
