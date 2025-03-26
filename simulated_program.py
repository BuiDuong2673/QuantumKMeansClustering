# Quantum tools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Classical tools
import numpy as np
import math
import random
import ast

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus

def polar_angle(point):
    """
    Calculate polar coordinate
    Args:
        point (list[float, float]): x and y cooridnate of a 2D point
    """
    angle = math.atan2(point[1], point[0])
    return angle

def polar_radius(point):
    """
    Calculate polar coordinate
    Args:
        point (list[float, float]): x and y cooridnate of a 2D point
    """
    radius = math.sqrt(point[0] ** 2 + point[1] ** 2)
    return radius

def initialize_clusters(k, dataset):
    """
    Initialize clusters as a dictionary where the keys are centroid coordinates
    and the values are lists of points in the clusters.
    Args:
        k (int): the number of clusters.
        dataset (list[list[float, float]]): List of points for the clustering 
            problem.
    """
    clusters = {}
    # ----------- OPTIONS FOR CHOOSING INITIAL CENTROIDS ---------------------
    # Choose k centroids using k-means++ method
    dataset_array = np.array(dataset)
    centroids, _ = kmeans_plusplus(dataset_array, n_clusters=k, random_state=0)
    centroids = centroids.tolist()
    # Choose k centroids randomly
    # centroids = random.sample(dataset, k)
    # ------------------------------------------------------------------------
    # Create a dictionary to store points in the cluster represented by centroids
    for centroid in centroids:
        clusters[f"{centroid}"] = []
    return clusters, centroids

def recompute_centroids(old_clusters, old_centroids):
    """
    Recompute centroids as the average of values in the clusters
    Args:
        old_clusters (dict): keys are centroids' coordinates and values are lists
            of points in the clusters
        old_centroids (list[float, float]): centroid coordinates [x, y]
    """
    # Define variables
    is_equal = True
    new_clusters = {}
    new_centroids = []
    for index, (_, cluster) in enumerate(old_clusters.items()):
        new_centroid = np.mean(cluster, axis=0).tolist()
        # Check if the differences between the new and old centroids are tolerable
        if np.linalg.norm(np.array(new_centroid) - np.array(old_centroids[index])) > 1e-4:
            is_equal = False
        # Prepare new clusters dictionary and centroids list
        new_clusters[f"{new_centroid}"] = []
        new_centroids.append(new_centroid)
    if is_equal is False:
        return new_clusters, new_centroids, is_equal
    return old_clusters, old_centroids, is_equal

def quantum_circuit(qc, qr, cr, i, data_angle, centroid_angle):
    """Prepare quantum circuit for calculate distance between a pair of points
    Args:
        qc (QuantumCircuit), qr (Quantum Register), cr (ClassicalRegister)
        i (int): number of iterations executed
        data_angle (float): polar angle of a data point
        centroid_angle (float): polar angle of a centroid
    """
    qc.h(qr[i * 2])
    qc.cx(qr[i * 2], qr[i * 2 + 1])
    qc.ry(-abs(data_angle - centroid_angle), qr[i * 2 + 1])
    qc.cx(qr[i * 2], qr[i * 2 + 1])
    qc.ry(abs(data_angle - centroid_angle), qr[i * 2 + 1])
    # Inteferernce and measurement
    qc.h(qr[i * 2])
    qc.measure(qr[i * 2], cr[i])
    return qc

def nearest_centroids_dictionary(dataset):
    """
    Construct a dictionary to store the information about nearest centroid for
    each point in the dataset
    Args:
        dataset (list[list[float, float]]): list of data points
    """
    nearest_centroids_dict = {}
    for data_point in dataset:
        nearest_centroids_dict[f"{data_point}"] = {
            'smallest distance': float('inf'),
            'nearest centroid': None
        }
    return nearest_centroids_dict

def classical_distance(point_1, point_2):
    """
    Classically calculate the Euclidean distance between 2 points
    """
    euclidean_distance = math.sqrt((point_1[0] - point_2[0]) ** 2 + 
                                   (point_1[1] - point_2[1]) ** 2)
    return euclidean_distance

def classical_selection(dataset, centroids):
    """Select nearest centroids based on the Euclidean distances computed classically
    Args:
        dataset (list[list[float, float]]): list of data points
        centroids (list[list[float, float]]): list of initial centroids
    """
    nearest_centroids_dict = nearest_centroids_dictionary(dataset)
    for data in dataset:
        for centroid in centroids:
            classical_dist = classical_distance(data, centroid)
            if classical_dist < nearest_centroids_dict[f"{data}"]['smallest distance']:
                nearest_centroids_dict[f"{data}"]['smallest distance'] = classical_dist
                nearest_centroids_dict[f"{data}"]['nearest centroid'] = centroid
    return nearest_centroids_dict

def quantum_distance(is_new_formula, calculated_distances,
                     nearest_centroids_dict, num_shots=1024):
    """Calculate Euclidean distances based on quantum computer's output
    Args:
        is_new_formula (bool): = True if calling our new formula, = False if 
            calling Khan et al. formula
        calculated_distances (dict): stores pairs of distances assigned to 
            quantum circuit
        nearest_centroids_dict (dict): stores nearest centroid for each data point
        num_shots (int): The number of shots the quantum circuit was executed
    """
    for _, item in calculated_distances.items():
        probability_1 = item['count 1'] / num_shots
        cen = item['pair'][0]
        data = item['pair'][1]
        if is_new_formula is True:
            data_radius, centroid_radius = polar_radius(data), polar_radius(cen)
            quantum_dist = math.sqrt((data_radius - centroid_radius) ** 2 +
                                     4 * data_radius * centroid_radius * probability_1)
        else: # Use Khan et al. formula
            normalize_value = math.sqrt(data[0] ** 2 + data[1] ** 2 + cen[0] ** 2 + cen[1] ** 2)
            quantum_dist = normalize_value * math.sqrt(2 * probability_1)
        # Update the nearest centroids dictionary
        if quantum_dist < nearest_centroids_dict[f"{data}"]['smallest distance']:
            nearest_centroids_dict[f"{data}"]['smallest distance'] = quantum_dist
            nearest_centroids_dict[f"{data}"]['nearest centroid'] = cen
    return nearest_centroids_dict

def quantum_selection(is_new_formula, dataset, initial_centroids, num_qubits, num_shots=1024):
    """ Choose nearest centroids based on Euclidean distances calculated quantumly
    Args:
        is_new_formula (bool): = True if calling our new formula, = False if 
            calling Khan et al. formula
        dataset (list[list[float, float]]): list of data points
        initial_centroid (list[list[float, float]]): list of initial centroids
        num_qubits (int): the number of qubits available on quantum circuit
        num_shots (int): The number of shots the quantum circuit was executed
    """
    # Calculate the number of distances can be calculated at the same time
    num_distances = 0
    num_distances, _ = divmod(num_qubits, 2)
    # Initialize the dictionary stores nearest centroids for each data point
    nearest_centroids_dict = nearest_centroids_dictionary(dataset)
    # Initialize a list contains all pairs of points in the data point
    pairs = []
    for centroid in initial_centroids:
        for data in dataset:
            pairs.append([centroid, data])
    # Loop until all pairs' distances are assigned to quantum circuit
    while len(pairs) > 0:
        # Initialize quantum circuit
        qr = QuantumRegister(num_distances * 2, name='q')
        cr = ClassicalRegister(num_distances, name='c')
        qc = QuantumCircuit(qr, cr)
        # Initialize dictionary contains all pairs of points whose distances are going to be calculated
        calculated_distances = {}
        i = 0
        # Loop until the circuit is full
        while (len(calculated_distances) < num_distances) and (len(pairs) > 0):
            pair = pairs.pop()
            centroid, data_point = pair
            centroid_angle, data_angle = polar_angle(centroid), polar_angle(data_point)
            qc = quantum_circuit(qc, qr, cr, i, data_angle, centroid_angle)
            # Update the index and calculated_distances
            calculated_distances[i] = {'pair': pair, 'count 1': 0}
            i += 1
        # Run quantum circuit
        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=num_shots)
        result = job.result()[0]
        count = result.data.c.get_counts()
        state_list = list(count.keys())
        # Count the number of time measured |1> for each pair of distance calculation
        for state in state_list:
            state_count = count.get(state, 0)
            for idx, bit in enumerate(state):
                if bit == '1':
                    if (len(calculated_distances) - idx - 1) >= 0:
                        calculated_distances[len(calculated_distances) - idx - 1]['count 1'] += state_count
        nearest_centroids_dict = quantum_distance(is_new_formula, calculated_distances,
                                                  nearest_centroids_dict, num_shots)
    return nearest_centroids_dict

def quantum_k_means(is_new_formula, k, dataset, clusters, centroids, num_qubits,
                    max_iteration=10, num_shots=1024):
    """Perform k-means clustering using quantum computer
    Args:
        is_new_formula (bool): = True if calling our new formula, = False if 
            calling Khan et al. formula
        k (int): the number of clusters
        dataset (list[list[float, float]]): list of data points
        clusters (dict): initial clusters with keys are initial
            centroids and values are empty list
        centroids (list[list[float, float]]): list of initial centroids
        num_qubits (int): number of available qubits
        max_iteration (int): maximum times the algorithm iterates if it still
            does not reach convergence
        num_shots (int): The number of shots the quantum circuit was executed
    """
    is_convergence = False
    iteration_count = 0
    method = "Our new" if is_new_formula else "Khan et al."
    file_name = "our_formula_output.txt" if is_new_formula else "Khan_formula_output.txt"
    message_to_txt = ""
    # Loop until convergence
    while (not is_convergence) and (iteration_count < max_iteration):
        iteration_count += 1
        # Run quantum program
        nearest_centroids_dict = quantum_selection(is_new_formula, dataset, centroids, num_qubits, num_shots)
        classical_dict = classical_selection(dataset, centroids)
        # Compare quantum choice with classical choice
        same_count, different_count = 0, 0
        for data_string, data_dict in nearest_centroids_dict.items():
            if data_dict['nearest centroid'] == classical_dict[data_string]['nearest centroid']:
                same_count += 1
            else:
                different_count += 1
            # Read and update the information to the clusters dictionary
            data = ast.literal_eval(data_string)
            nearest_centroid = data_dict['nearest centroid']
            clusters[f"{nearest_centroid}"].append(data)
        message_1 = f"{method} formula's same count: {same_count} and different count: {different_count}\n"
        message_to_txt += message_1
        print(message_1)
        if any(len(cluster) == 0 for cluster in clusters.values()):
            print("Empty cluster detected. Reinitializing centroids.")
            clusters, centroids = initialize_clusters(k, dataset)
            continue
        # Recompute centroids by averaging all points
        new_clusters, new_centroids, is_convergence = recompute_centroids(clusters, centroids)
        if is_convergence is True:
            break
        if iteration_count == max_iteration:
            print(f"\nNot converged but reached the max number of iteration {max_iteration}")
            break
        clusters, centroids = new_clusters, new_centroids
    message_2 = f"{method} formula's iteration count is: {iteration_count}\n"
    message_to_txt += message_2
    write_to_txt(message_to_txt, file_name)
    print(message_2)
    print(f"This output is stored in {file_name}")
    return clusters

def plot_clusters(ax, clusters, initial_centroids, title):
    """Subprogram to plot clusters from both formula and correct clusters in a
    horizontal row.
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot the clusters on
        clusters (dict): keys are centroids and values are list of points in cluster
        initial_centroids (list[list[float, float]]): the initial centroids before clustering
        title (str): the title of the plot
    """
    ax.scatter(initial_centroids[0], initial_centroids[1], marker='o', s=100, color='black')
    for centroid_string, cluster in clusters.items():
        # Plot centroids
        centroid = ast.literal_eval(centroid_string)
        ax.scatter(centroid[0], centroid[1], marker='x', s=100, color='black')
        # Plot points
        x_values = []
        y_values = []
        for point in cluster:
            x_values.append(point[0])
            y_values.append(point[1])
        ax.scatter(x_values, y_values)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)

def write_to_txt(message, file_name):
    """Write a message to .txt file
    Args:
        message (str): message to be written into the .txt file
        file_name (str): name of the .txt file
    """
    with open(f"{file_name}", "w", encoding="utf-8") as file:
        file.write(message)

def main():
    # Import Iris dataset
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    # Apply MinMaxScaler to map data onto (0, 1)
    features = MinMaxScaler().fit_transform(features)
    # Reduce the number of features
    features = PCA(n_components=2).fit_transform(features)
    # Change features to list and create list which store correct labels
    data_list = features.tolist()
    correct_list = labels.tolist()
    # Create a clusters dictionary from the information obtained above
    correct_clusters = {0: [], 1: [], 2: []}
    for i, label in enumerate(correct_list):
        if label == 0:
            correct_clusters[0].append(data_list[i])
        elif label == 1:
            correct_clusters[1].append(data_list[i])
        else:
            correct_clusters[2].append(data_list[i])
    # Change the dictionary keys to centroids
    new_correct_clusters = {}
    for i, cluster in correct_clusters.items():
        centroid = np.mean(cluster, axis=0).tolist()
        new_correct_clusters[f"{centroid}"] = cluster
    correct_clusters = new_correct_clusters

    # Run the experiment
    k = 3 # the number of clusters
    num_qubits = 10
    max_iteration = 10
    num_shots = 2048

    # Initialize empty clusters with randomly chosen centroids
    clusters, centroids = initialize_clusters(k, data_list)
    print(f"Original centroids {centroids}")
    our_clusters = quantum_k_means(True, k, data_list, clusters, centroids, num_qubits, max_iteration, num_shots)
    khan_clusters = quantum_k_means(False, k, data_list, clusters, centroids, num_qubits, max_iteration, num_shots)
    
    # Write the clusters to .txt file
    write_to_txt(str(our_clusters), "new_formula_clusters.txt")
    write_to_txt(str(khan_clusters), "Khanetal_clusters.txt")
    write_to_txt(str(correct_clusters), "correct_clusters.txt")

    # Plot the clusters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plot_clusters(axes[0], khan_clusters, centroids, 'Khan et al. method result')
    plot_clusters(axes[1], our_clusters, centroids, 'Our method result')
    plot_clusters(axes[2], correct_clusters, centroids, 'Correct labels')
    fig.tight_layout()
    image_name = "clustering_results.png"
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    print(f"The plot is saved as image with name: {image_name}")

if __name__ == "__main__":
    main()
