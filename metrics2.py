# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric

def kmeans_minkowski(X, n_clusters, p=2, random_state=42):
    """
    Aplica K-means con distancia de Minkowski
    p=1: Distancia de Manhattan
    p=2: Distancia Euclidiana (por defecto)
    p=inf: Distancia de Chebyshev
    """
    # Crear métrica de distancia personalizada
    minkowski_metric = DistanceMetric.get_metric('minkowski', p=p)
    
    # Aplicar K-means con la métrica personalizada
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    
    return kmeans

# Generar datos sintéticos de tarjetas de crédito
np.random.seed(42)
n_samples = 1000

# Crear características
saldo = np.concatenate([
    np.random.normal(1000, 500, n_samples//3),  # Saldo bajo
    np.random.normal(5000, 1500, n_samples//3),  # Saldo medio
    np.random.normal(10000, 2000, n_samples//3)  # Saldo alto
])

frecuencia_compras = np.concatenate([
    np.random.normal(5, 2, n_samples//3),    # Frecuencia baja
    np.random.normal(15, 5, n_samples//3),   # Frecuencia media
    np.random.normal(30, 8, n_samples//3)    # Frecuencia alta
])

# Crear DataFrame
X = np.column_stack((saldo, frecuencia_compras))
df = pd.DataFrame(X, columns=['Saldo', 'Frecuencia_Compras'])

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar el número óptimo de clusters usando el método del codo
inertias = []
k_range = range(2, 8)

# Probar diferentes valores de p para la distancia de Minkowski
p_values = [1, 2, 3, 5]  # Manhattan, Euclidiana, y otros valores
minkowski_results = {}

for p in p_values:
    inertias_p = []
    for k in k_range:
        kmeans = kmeans_minkowski(X_scaled, k, p=p)
        inertias_p.append(kmeans.inertia_)
    minkowski_results[f'p={p}'] = inertias_p

# Visualizar el método del codo
plt.figure(figsize=(12, 8))

# Crear subgráficas para diferentes valores de p
for i, p in enumerate(p_values):
    plt.subplot(2, 2, i+1)
    plt.plot(k_range, minkowski_results[f'p={p}'], 'o-', label=f'p={p}')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title(f'Método del Codo - Distancia de Minkowski (p={p})')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig('minkowski_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica de análisis de Minkowski guardada como 'minkowski_analysis.png'")

# Aplicar K-means con el número óptimo de clusters (k=3) usando diferentes distancias de Minkowski
print("Comparación de clustering con diferentes distancias de Minkowski:")
print("=" * 60)

for p in p_values:
    kmeans_final = kmeans_minkowski(X_scaled, 3, p=p)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    print(f"\nDistancia de Minkowski con p={p}:")
    print(f"Inercia: {kmeans_final.inertia_:.2f}")
    
    # Analizar los clusters
    for i in range(3):
        cluster_size = np.sum(clusters == i)
        avg_saldo = np.mean(X[clusters == i, 0])
        avg_freq = np.mean(X[clusters == i, 1])
        print(f"  Cluster {i}: {cluster_size} clientes, Saldo: ${avg_saldo:.2f}, Frecuencia: {avg_freq:.2f}")

# Usar p=2 (Euclidiana) para la visualización principal
kmeans_final = kmeans_minkowski(X_scaled, 3, p=2)
clusters = kmeans_final.fit_predict(X_scaled)

# Visualizar los resultados
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(
    kmeans_final.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
    kmeans_final.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
    marker='x', s=200, linewidths=3, color='r', label='Centroides'
)
plt.title('Segmentación de Clientes de Tarjetas de Crédito (Minkowski p=2)')
plt.xlabel('Saldo ($)')
plt.ylabel('Frecuencia de Compras (mensual)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.savefig('minkowski_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica de clusters con Minkowski guardada como 'minkowski_clusters.png'")
