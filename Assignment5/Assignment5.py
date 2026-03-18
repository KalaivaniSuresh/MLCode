import matplotlib
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#from kneed import KneeLocator
import os

# ================== SETUP ==================
base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

def save_plot(filename):
    plt.savefig(os.path.join(images_dir, filename))


# ================== VISUALIZATION ==================
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()


# ================== TASK 1 ==================
def get_ground_level(pcd, dataset_name):
    z_vals = pcd[:, 2]

    hist, bin_edges = np.histogram(z_vals, bins=100)
    max_bin_index = np.argmax(hist)

    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    plt.figure(figsize=(8,5))
    plt.hist(z_vals, bins=100, alpha=0.7)
    plt.axvline(ground_level, linestyle='--',
                label=f'Ground = {ground_level:.2f}')

    plt.title(f"Histogram (Z values) - {dataset_name}")
    plt.xlabel("Z values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()

    save_plot(f"histogram_{dataset_name}.png")
    plt.show()

    return ground_level


# ================== TASK 2 ==================
def run_dbscan_with_elbow(pcd, dataset_name, k=10, eps_value=None):

    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(pcd)
    distances, _ = nbrs.kneighbors(pcd)

    k_distances = np.sort(distances[:, k-1])

    # Elbow plot
    plt.figure(figsize=(8,5))
    plt.plot(k_distances)
    plt.title(f"Elbow Plot - {dataset_name}")
    plt.xlabel("Points sorted")
    plt.ylabel("Distance")
    plt.grid()

    save_plot(f"elbow_{dataset_name}.png")
    plt.show()

    if eps_value is None:
       print(f"[{dataset_name}] 👉 Choose eps from elbow plot")
       return None

    print(f"[{dataset_name}] Using eps =", eps_value)

    # Automatically calculate eps if not provided
    # if eps_value is None:
    #     # Use KneeLocator to detect the elbow
    #     knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
    #     eps_value = k_distances[knee.knee]
    #     print(f"[{dataset_name}] Calculated optimal eps from elbow:", eps_value)
    # else:
    #     print(f"[{dataset_name}] Using manual eps =", eps_value)
    
    #DB Scan clustering
    clustering = DBSCAN(eps=eps_value, min_samples=k).fit(pcd)
    labels = clustering.labels_

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[{dataset_name}] Number of clusters:", clusters)

    # Cluster plot
    plt.figure(figsize=(10,10))
    plt.scatter(pcd[:,0], pcd[:,1], c=labels, cmap='tab20', s=2)

    plt.title(f"Clusters - {dataset_name} (clusters={clusters})")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    save_plot(f"clusters_{dataset_name}.png")
    plt.show()

    return labels


# ================== TASK 3 ==================
def extract_catenary_cluster(pcd, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    best_cluster = None
    best_span = 0
    best_bounds = None

    for label in unique_labels:
        cluster_points = pcd[labels == label]

        x_min, y_min = np.min(cluster_points[:, :2], axis=0)
        x_max, y_max = np.max(cluster_points[:, :2], axis=0)

        span = (x_max - x_min) + (y_max - y_min)

        if span > best_span:
            best_span = span
            best_cluster = cluster_points
            best_bounds = (x_min, y_min, x_max, y_max)

    return best_cluster, best_bounds


# ================== PIPELINE ==================
def process_dataset(file_path, dataset_name, eps_value):

    print(f"\n========== {dataset_name} ==========\n")

    pcd = np.load(file_path)

    # Visualize
    show_cloud(pcd)

    # ---- Task 1 ----
    ground_level = get_ground_level(pcd, dataset_name)
    print("Ground level:", ground_level)

    pcd_above_ground = pcd[pcd[:,2] > ground_level + 3.0]

    print("Original size:", pcd.shape)
    print("Above ground size:", pcd_above_ground.shape)

    show_cloud(pcd_above_ground)

    # ---- Task 2 (Elbow) ----
    run_dbscan_with_elbow(pcd_above_ground, dataset_name, k=5)

    # ---- Task 2 (DBSCAN) ----
    labels = run_dbscan_with_elbow(
        pcd_above_ground,
        dataset_name,
        k=5,
        eps_value=eps_value
    )

    # ---- Task 3 ----
    catenary_cluster, bounds = extract_catenary_cluster(
        pcd_above_ground,
        labels
    )

    print("Catenary bounds (min x, min y, max x, max y):")
    print(bounds)

    # Plot catenary
    plt.figure(figsize=(8,8))
    plt.scatter(catenary_cluster[:,0], catenary_cluster[:,1], s=2)

    plt.title(f"Catenary - {dataset_name}")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    save_plot(f"catenary_{dataset_name}.png")
    plt.show()

    return ground_level, bounds


# ================== MAIN ==================

dataset1_path = r"C:\Users\Hp\OneDrive - Luleå University of Technology\Master\LP3\Introduction to Industrial AI and eMaintenance D7015B 35654\Assignment 5\Lidar_assignment-1\dataset1.npy"

dataset2_path = r"C:\Users\Hp\OneDrive - Luleå University of Technology\Master\LP3\Introduction to Industrial AI and eMaintenance D7015B 35654\Assignment 5\Lidar_assignment-1\dataset2.npy"


# -------- Run Dataset 1 --------
ground1, bounds1 = process_dataset(
    dataset1_path,
    "dataset1",
    eps_value=2.5   # your tuned value
)


# -------- Run Dataset 2 --------
ground2, bounds2 = process_dataset(
    dataset2_path,
    "dataset2",
    eps_value=2.7   # ⚠️ adjust after elbow
)