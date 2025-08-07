import pandas as pd
url = r"C:\Users\samar kumar\Downloads\kaggle\Iris.csv"
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv(url)


print(iris.describe())

# Visualizing data distribution.
import matplotlib.pyplot as plt
iris['SepalLengthCm'].hist()
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length')
plt.show()

# Loading the dataset into a Pandas DataFrame
iris = pd.read_csv(url, header=None, names=col_names)

# Mapping the classes to colors for visualization
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot for Sepal Length vs. Sepal Width
for species, color in colors.items():
    subset = iris[iris['class'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=species, color=color, alpha=0.7)

# Adding labels and title
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.legend(title='Species')
plt.grid(True)
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset into a Pandas DataFrame
iris = pd.read_csv(url)

# Separate the features and the target
X = iris.iloc[:, :-1]
y = iris['Species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['class'] = y

# Define colors for each class
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

# Plot the PCA representation
plt.figure(figsize=(10, 6))
for species, color in colors.items():
    subset = pca_df[pca_df['class'] == species]
    plt.scatter(subset['PC1'], subset['PC2'], label=species, color=color, alpha=0.7)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend(title='Species')
plt.grid(True)
plt.show()