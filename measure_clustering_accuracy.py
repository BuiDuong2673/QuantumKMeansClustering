import ast

def read_data_from_txt(file_name):
    """Open and read clusters dictionary from the data file"""
    # Open and read clusters dictionary from the data files
    with open(file_name, 'r', encoding="utf-8") as file:
        clusters_string = file.read()
        clusters = ast.literal_eval(clusters_string)
    return clusters

def read_centroids(clusters):
    """Read centroids from the clusters dictionary"""
    centroids = []
    for centroid_string, _ in clusters.items():
        centroids.append(centroid_string)
    return centroids

def print_centroids(centroids, quantum_method):
    """Print centroids information to support next actions"""
    print(f"{quantum_method}'s centroids are:\n")
    for i, centroid in enumerate(centroids):
        print(f"({i + 1}) {centroid}")

def ask_for_input(correct_centroids, quantum_method):
    """Collect user inputs to find the quantum cluster corresponds to correct cluster"""
    quantum_indexes = []
    for i, correct_centroid in enumerate(correct_centroids):
        while True:
            try:
                quantum_input = int(input(f"Which {quantum_method}'s centroid corresponds to correct centroid ({i + 1}) {correct_centroid}?\nCentroid (1, 2, or 3): "))
                quantum_index = quantum_input - 1
                if quantum_index not in [0, 1, 2]:
                    print("Please enter only 1, 2, or 3!")
                    continue
                if quantum_index in quantum_indexes:
                    print("Please do not enter a value twice!")
                    continue
                quantum_indexes.append(quantum_index)
                break
            except ValueError:
                print("Invalid input! Please enter a number (1, 2, or 3).")
    return quantum_indexes

def measure_accuracy(quantum_indexes, quantum_centroids, quantum_clusters, correct_centroids, correct_clusters, quantum_method):
    """Measure the accuracy of quantum method"""
    accurate_points = 0
    wrong_points = 0
    for i in range(3):
        for point in quantum_clusters[quantum_centroids[quantum_indexes[i]]]:
            if point in correct_clusters[correct_centroids[i]]:
                accurate_points += 1
            else:
                wrong_points += 1
    accurate_rate = (accurate_points / (accurate_points + wrong_points)) * 100
    print(f"{quantum_method} formula accuracy is: {accurate_rate}%.")

def main():
    khan_clusters = read_data_from_txt("Khanetal_clusters.txt")
    new_method_clusters = read_data_from_txt("new_formula_clusters.txt")
    correct_clusters = read_data_from_txt("correct_clusters.txt")

    khan_centroids = read_centroids(khan_clusters)
    new_method_centroids = read_centroids(new_method_clusters)
    correct_centroids = read_centroids(correct_clusters)

    print_centroids(khan_centroids, "Khan et al.")
    print_centroids(new_method_centroids, "Our method")
    print_centroids(correct_centroids, "Correct")

    khan_indexes = ask_for_input(correct_centroids, "Khan et al.")
    new_method_indexes = ask_for_input(correct_centroids, "Our method")

    print(f"Khan indexes {khan_indexes}")
    print(f"New method indexes {new_method_indexes}")

    measure_accuracy(khan_indexes, khan_centroids, khan_clusters, correct_centroids, correct_clusters, "Khan et al.")
    measure_accuracy(new_method_indexes, new_method_centroids, new_method_clusters, correct_centroids, correct_clusters, "Our method")

if __name__ == "__main__":
    main()
