import matplotlib.pyplot as plt

def count_same_different(file_name, is_new_method):
    """read same, different counts data from the output of the main program
    stored in .txt file"""
    same_counts = []
    different_counts = []
    if is_new_method:
        same_index = 5
        different_index = 9
    else:
        same_index = 6
        different_index = 10
    # Open and read data in data files
    with open(file_name, 'r', encoding="utf-8") as file:
        for line in file:
            if "formula's same count:" in line and "and different count:" in line:
                # Split the line by spaces
                parts = line.split()
                # Extract the same counts and different counts
                same_count = int(parts[same_index])
                different_count = int(parts[different_index])
                # Append the counts to lists
                same_counts.append(same_count)
                different_counts.append(different_count)
    return same_counts, different_counts

def plot_bar_chart(same_counts, different_counts, is_new_formula, image_name):
    """Create bar chart showing same counts and different counts"""
    method = "Our new" if is_new_formula else "Khan et al."
    # Add labels for each column
    labels = []
    for i in range(len(same_counts)):
        labels.append(f"{i + 1}")
    # Create the plot
    _, ax = plt.subplots()
    ax.bar(labels, same_counts, color='blue', label='Same')
    ax.bar(labels, different_counts, bottom=same_counts, color='red', label='Different')
    # Add labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Same/ Different Count')
    ax.set_title(f"{method} formula-Classical selected centroids comparisons")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight')

khan_same, khan_different = count_same_different("Khan_formula_output.txt", False)
our_same, our_different = count_same_different("our_formula_output.txt", True)
plot_bar_chart(khan_same, khan_different, False, "KhanPlot.png")
plot_bar_chart(our_same, our_different, True, "OurPlot.png")
