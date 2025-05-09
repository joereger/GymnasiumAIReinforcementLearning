# Joe Reger's Reinforcement Learning Experiments with Gymnasium

This repository contains a collection of experiments and solutions for various reinforcement learning problems using the [Gymnasium](https://gymnasium.farama.org/) library by the Farama Foundation. The primary goal is personal learning and exploration of different RL algorithms and techniques.

## Project Structure

The project is organized with a dedicated folder for each Gymnasium environment being tackled. Each environment's folder aims to be self-contained, including all solution scripts and necessary utility files.

-   `bipedal_walker/`: Solutions for the Bipedal Walker environment.
-   `cart_pole/`: Solutions for the Cart Pole environment.
-   `lunar_lander/`: Solutions for the Lunar Lander environment.
-   `mountain_car/`: Solutions for the Mountain Car environment.
-   `memory-bank/`: Contains detailed documentation for Cline (AI assistant) to maintain context about the project, including specifics for each environment.
-   `data/`: (Optional) May contain saved models, logs, or experiment results.
-   `GymnasiumVENV/`: Python virtual environment for the project.

## Approaches Applied (Summary)

Detailed information about the specific algorithms, hyperparameters, and results for each approach can be found in the `approaches.md` file within each environment's Memory Bank directory (e.g., `memory-bank/environments/bipedal_walker/approaches.md`).

Here's a high-level overview of environments and the corresponding solution files:

*   **Bipedal Walker:**
    *   `bipedal_walker/bipedal_walker_plus_genetic_algorithm.py`
    *   `bipedal_walker/bipedal_walker-a3c.py`
    *   `bipedal_walker/bipedal_walker.py`
*   **Cart Pole:**
    *   `cart_pole/cart_pole.py`
*   **Lunar Lander:**
    *   `lunar_lander/lunar_lander.py` (currently a basic random agent script)
*   **Mountain Car:**
    *   `mountain_car/mountain_car_discrete.py` (currently a basic random agent script)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up Python Virtual Environment:**
    It's recommended to use the provided `GymnasiumVENV/` or create your own.
    To activate the existing venv (on macOS/Linux):
    ```bash
    source GymnasiumVENV/bin/activate
    ```
    If creating a new one:
    ```bash
    python3 -m venv MyVenv
    source MyVenv/bin/activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file will be provided. Install the necessary packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    This will install Gymnasium and other required libraries like PyTorch, TensorFlow (if used by specific agents), NumPy, and Matplotlib.

    Gymnasium itself can be installed with environment-specific extras if needed, e.g., `pip install "gymnasium[box2d]"`. Refer to the [Gymnasium documentation](https://gymnasium.farama.org/environments/box2d/) for details on specific environment dependencies.

## How to Run Experiments

Each environment's solution script can typically be run directly from its respective folder.

For example, to run a Bipedal Walker experiment:
```bash
cd bipedal_walker
python bipedal_walker.py
```

Or to run the Cart Pole experiment:
```bash
cd cart_pole
python cart_pole.py
```

Ensure your virtual environment is activated before running the scripts. Some scripts might save models or logs to the `data/` directory or subdirectories within their environment folder.

## Gymnasium Version

This project aims to use the Farama Foundation's `gymnasium` library. The code will be reviewed and updated to ensure compatibility with `gymnasium` (currently targeting v1.1.1 or similar recent versions). The `import gymnasium as gym` convention is used.
