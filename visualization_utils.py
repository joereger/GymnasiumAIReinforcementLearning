# mah little helper utils for visualizing activations

import matplotlib.pyplot as plt
import numpy as np

def plot_activations(activations):
    print("Plotting starts")
    plt.clf()  # Clear the previous plot
    prev_x = 0
    for layer_name, activation in activations.items():
        norm_activation = np.clip(activation.detach().numpy(), 0, 1)
        print("Min:", np.min(norm_activation))
        print("Max:", np.max(norm_activation))
        print("Mean:", np.mean(norm_activation))
        plot_layer(prev_x + 20, norm_activation, prev_x, prev_activation if 'prev_activation' in locals() else None)
        prev_x += 40
        prev_activation = norm_activation
    
    plt.draw()
    plt.pause(0.001)
    plt.show()  # Explicitly show the plot
    print("Plotting ends")

def plot_layer(x, activations, prev_x=None, prev_activations=None):
    for i, activation in enumerate(activations):
        y = i * 10
        # Check if activation is an array and take the average if so
        if isinstance(activation, np.ndarray):
            activation = np.mean(activation)
        # Ensure activation is between 0 and 1
        norm_activation = min(max(activation, 0), 1)
        circle = plt.Circle((x, y), 3, color=plt.cm.viridis(norm_activation))
        plt.gca().add_patch(circle)
        if prev_x is not None and prev_activations is not None:
            prev_y = i * 10
            line = plt.Line2D([prev_x + 3, x - 3], [prev_y, y], linewidth=1, color='grey')
            plt.gca().add_line(line)


def register_hooks(model, activations):
    for name, layer in model.named_children():
        layer.register_forward_hook(lambda module, input, output, name=name: activations.update({name: output}))
