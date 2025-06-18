# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos de cáncer de Wisconsin
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# Evaluar el rendimiento
from sklearn.metrics import adjusted_rand_score, silhouette_score

print("Puntuación Rand Ajustada:", adjusted_rand_score(y, y_pred))
print("Coeficiente de Silueta:", silhouette_score(X_scaled, y_pred))

# Visualizar los resultados en las dos primeras características
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker='x', s=200, linewidths=3, color='r', label='Centroides')
plt.title('K-means con Dataset de Cáncer de Wisconsin')
plt.xlabel('Primera característica estandarizada')
plt.ylabel('Segunda característica estandarizada')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

# Imprimir información sobre los clusters
for i in range(2):
    print(f"\nCluster {i}:")
    print(f"Número de muestras: {np.sum(y_pred == i)}")
    print(f"Porcentaje de casos malignos: {np.mean(y[y_pred == i]) * 100:.2f}%")
