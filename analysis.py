import pandas as pd
import matplotlib.pyplot as plt


# 1. Load the Iris dataset
df = pd.read_csv('/mnt/c/Users/joshu/Downloads/Iris.csv')


# 2. Display the first five rows
print("First five rows:")
print(df.head())


# 3. Check the structure of the dataset
print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nData types:")
print(df.dtypes)


# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# 4. Summary statistics

# Mean
print("\nMean:")
print(df.mean(numeric_only=True))

# Median
print("\nMedian:")
print(df.median(numeric_only=True))

# Minimum
print("\nMinimum:")
print(df.min(numeric_only=True))

# Maximum
print("\nMaximum:")
print(df.max(numeric_only=True))

# Standard deviation
print("\nStandard deviation:")
print(df.std(numeric_only=True))


# Display all summary statistics together
print("\nSummary statistics:")
print(df.describe())


# 5. Basic exploratory data analysis

# Count the number of flowers in each species
print("\nNumber of flowers in each species:")
print(df["Species"].value_counts())


# 6. Visualizations


# A. Line Chart
plt.figure(figsize=(8, 5))

plt.plot(df["Id"], df["SepalLengthCm"], label="Sepal Length")
plt.plot(df["Id"], df["PetalLengthCm"], label="Petal Length")

plt.title("Sepal Length and Petal Length")
plt.xlabel("Flower ID")
plt.ylabel("Length (cm)")
plt.legend()
plt.grid(True)
plt.savefig('line_chart.png')
plt.close()


# B. Scatter Plot
plt.figure(figsize=(8, 5))

plt.scatter(df["SepalLengthCm"], df["PetalLengthCm"])

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.close()


# C. Histogram
plt.figure(figsize=(8, 5))

plt.hist(df["SepalLengthCm"], bins=10, edgecolor="black")

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")

plt.savefig('histogram.png')
plt.close()


# D. Bar Chart
species_count = df["Species"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(species_count.index, species_count.values)

plt.title("Number of Flowers by Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.xticks(rotation=20)
plt.savefig('bar_chart.png')
plt.close()
