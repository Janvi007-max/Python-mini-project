import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("train_and_test2.csv")
print("First Five rows of dataset.")
print(df.head(), "\n")

# 2. Fix colummn name
df = df.rename(columns={"2urvived" : "Survived"})

# 3. Drop all zero columns
df = df.loc[:, ~df.columns.str.contains("zero")]

print("=== Cleaned Columns ====")
print(df.columns, "\n")

# 4. Finding missing values
print("==== Missing values (Before) ====")
print(df.isnull().sum(), "\n")

# Missing Values in age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Missing Values in fare
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Missing values in embarked
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("==== Missing Values (After) ====")
print(df.isnull().sum(), "\n")

# Fixing Sex column
df['Sex'] = df['Sex'].map({0 : "male", 1 : "female"})



# Basic Stats
print("==== Summary Stats =====")
print(df.describe(), "\n")

print("==== Value Count ====")
print("Sex: \n ", df['Sex'].value_counts(), "\n")
print("Embarked: \n ", df['Embarked'].value_counts(), "\n")
print("Survived: \n ", df['Survived'].value_counts(), "\n")

print("=== Corelation ===")
print(df.corr(numeric_only=True), "\n")

# Vizulization
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['Age'], bins=15, kde=True, color="skyblue")
plt.title("Age Distribution")

plt.subplot(2, 2, 2)
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set2")
plt.title("Survival by passanger class")

plt.subplot(2, 2, 3)
sns.countplot(x="Sex", hue="Survived", data=df, palette="pastel")
plt.title("Survival by Sex")

plt.subplot(2, 2, 4)
sns.scatterplot(x="Age", y="Fare", hue= "Survived", data=df, palette="coolwarm", alpha=0.7)
plt.title("Age vs Fare (Coloured by Survuval)")

plt.tight_layout()
plt.savefig("Insights.png")

print("==== Key Insights ====")
print(f"Average Age: {df['Age'].mean():.2f}")
print(f"Average Fare: {df['Fare'].mean():.2f}")
print(f"Overall Survival Rate: {df['Survived'].mean() * 100}")
