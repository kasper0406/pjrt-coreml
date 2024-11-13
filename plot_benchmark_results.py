import numpy as np
import matplotlib.pyplot as plt

def plot_power():
    # Define the data
    labels = ['CPU', 'ANE', 'GPU']
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    # CoreML data
    coreml_means = [4.356370000000001, 7.171180000000001, 0.02194]
    coreml_stds = [0.07671914363442796, 0.07190425995725222, 0.004146070428731282]

    # JAX-Metal data
    jaxmetal_means = [3.075845, 0.0, 104.02717499999999]
    jaxmetal_stds = [0.12328321601094006, 0.0, 3.671995922465037]

    # Create the plot
    fig, ax = plt.subplots()

    # Plot CoreML bars
    rects1 = ax.bar(x - width/2, coreml_means, width, yerr=coreml_stds, label='coreml', capsize=5)

    # Plot JAX-Metal bars
    rects2 = ax.bar(x + width/2, jaxmetal_means, width, yerr=jaxmetal_stds, label='jax-metal', capsize=5)

    # Add labels, title, and legend
    ax.set_ylabel('Power Consumption [J]')
    ax.set_title('Power Consumption - matmul and reduce')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


def plot_running_time():
    # Execution time data
    tests = ['coreml', 'jax-metal']
    x = np.arange(len(tests))  # label locations
    width = 0.5  # width of the bars

    # Mean execution times
    mean_times = [9.641074609756469, 3.1030665397644044]

    # Standard deviations
    std_times = [0.017313780001272098, 0.10156818762959695]

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the bars with error bars
    rects = ax.bar(x, mean_times, width, yerr=std_times, capsize=10, color=['skyblue', 'lightgreen'])

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time by Test')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)

    # Add value labels on top of each bar
    for rect, mean in zip(rects, mean_times):
        height = rect.get_height()
        ax.annotate(f'{mean:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plot_power()
plot_running_time()

plt.show()
