# NISQ Quantum Algorithm for K-Means Clustering

These scripts support the experiments presented in our research article:

Duong Bui and Kimmo Halunen. 2025. NISQ Quantum Algorithm for K-Means Clustering. In *The 40th ACM/SIGAPP Symposium on Applied Computing (SAC '25)*, March 31-April 4, 2025, Catania, Italy. ACM, New York, NY, USA, Article 4, 8 pages. https://doi.org/10.1145/3672608.3707726.

## Overview
The program includes:
* Main programs:
    - `simulated_program.py`: Runs experiments on quantum simulators.
    - `real_program.py`: Runs experiments on real IBM quantum systems.
* Measurement programs:
    - `measure_clustering_accuracy.py`: Measures the accuracy of quantum methods compared to the correct dataset labels.
    - `bar_plot.py`: Plots the number of times quantum methods select the same or different nearest centroids compared to the classical method.

## Requirements
* To run the program, install `Qiskit` by following this guide: https://docs.quantum.ibm.com/start/install#local.
* To run the program on a real IBM quantum computer:
    - Register for an account on the IBM Quantum Platform: https://quantum.ibm.com/
    - Copy the API token from the IBM Quantum Platform.
    - Paste the API token in `real_program.py`:
    ```python
    QiskitRuntimeService.save_account(channel="ibm_quantum",
                                      token="YOUR API TOKEN HERE",
                                      set_as_default=True, overwrite=True)
    ```

## Experiment Steps
* **Step 1**: Run `simulated_program.py` to execute on a quantum simulator, or `real_program.py` to execute on a real quantum system. The raw experiment results will be stored in:
    - `clustering_results.png`: Visualizations of both quantum and correct clusters.
    - `new_formula_clusters.txt`: Cluster results from our new formula.
    - `Khanetal_clusters.txt`: Cluster results from Khan et al.'s formula.
    - `our_formula_output.txt`: Counts of matches and mismatches between our formula and the classical method.
    - `Khan_formula_output.txt`: Counts of matches and mismatches between Khan et al.'s formula and the classical method.
    
    Apart from `clustering_results.png`, which is for visualization, the other files contain raw data for measurement programs.

* **Step 2**: Run `measure_clustering_accuracy.py` to evaluate the clustering accuracy of Khan et al.'s formula and our formula.
* **Step 3**: Run `bar_plot.py` to generate bar charts (`KhanPlot.png` and `OurPlot.png`) visualizing the same/different centroid selections between quantum and classical methods.

    Steps 2 and 3 can be performed in any order.

## Additional Information
### 1. Experiment Setup
Users can modify the following parameters in `simulated_program.py` (lines 349-352) and `real_program.py` (lines 362-365):
```python
k = 3  # Number of clusters
num_qubits = 10  # Number of qubits used in the quantum circuit
max_iteration = 10  # Maximum clustering iterations before stopping
num_shots = 2048  # Number of times the quantum circuit executes
```

### 2. Difference Between `simulated_program.py` and `real_program.py`
Both programs are nearly identical but differ in how they call quantum components.

In `simulated_program.py`, the quantum execution is straightforward:
```python
sampler = StatevectorSampler()
job = sampler.run([qc], shots=num_shots)
...
num_qubits = 10
```

In `real_program.py`, additional IBM Quantum Platform interactions are required:
```python
QiskitRuntimeService.save_account(channel="ibm_quantum",
                                  token="YOUR API TOKEN HERE",
                                  set_as_default=True, overwrite=True)
...
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
circuit = pm.run(qc)
sampler = Sampler(backend)
job = sampler.run([circuit], shots=num_shots)
...
num_qubits = 100
```

### 3. Choosing a Method to Initialize Centroids
In both `simulated_program.py` and `real_program.py`, the function `initialize_clusters(k, dataset)` allows two methods for centroid initialization:

* **(1) K-Means++ Initialization:**
```python
dataset_array = np.array(dataset)
centroids, _ = kmeans_plusplus(dataset_array, n_clusters=k, random_state=0)
centroids = centroids.tolist()
```

* **(2) Random Initialization:**
```python
centroids = random.sample(dataset, k)
```
Uncomment the preferred method and comment out the other to choose.