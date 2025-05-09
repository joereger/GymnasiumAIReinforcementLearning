# System Patterns: Gymnasium Environment Solutions

**Overall Repository Structure:**
- The project root contains dedicated directories for each Gymnasium environment (e.g., `bipedal_walker/`, `cart_pole/`). Each such directory holds all Python scripts related to solving that environment, including solution approaches and any necessary utility scripts.
- A structured `data/` directory with environment-specific subdirectories (e.g., `data/cart_pole/`, `data/lunar_lander/`) is used for storing experiment results, trained models, and other environment-specific artifacts. This ensures data isolation between environments.
- A `GymnasiumVENV/` directory houses the Python virtual environment for the project, ensuring consistent dependencies.
- The `memory-bank/` directory contains all documentation for Cline's Memory Bank.
- Obsolete Python scripts previously at the root level (e.g., `bipedal_walker.py`, `visualization_utils.py`) have been moved into their respective environment folders or will be removed.

**Memory Bank Structure:**
- **Top-Level Files:** Located directly in `memory-bank/`, these files provide overall project context (brief, product context, general system patterns, overall tech context, active project-level work, and overall progress).
- **`memory-bank/environments/` Directory:** This directory contains subfolders for each specific Gymnasium environment being worked on.
    - **`memory-bank/environments/<environment_name>/`:** Each subfolder holds a set of Memory Bank files (`environment_brief.md`, `approaches.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`) dedicated to that particular environment, ensuring isolated and detailed context.

**Code Organization within Environment Solutions:**
- Each environment (e.g., `bipedal_walker/`) has its own dedicated directory at the project root.
- All Python scripts for an environment, including different solution approaches (e.g., `bipedal_walker/bipedal_walker-a3c.py`) and any utility scripts (e.g., `bipedal_walker/visualization_utils.py`), reside within that environment's directory.
- **Self-Containment Principle:** Each environment's code is self-contained. Utility functions or helper scripts are duplicated within each environment's folder if needed, rather than relying on a shared root-level utilities folder. This ensures each environment can be understood and run independently.
- **Data Storage Principle:** All environment-specific data (models, logs, results) is stored in environment-specific data subdirectories (e.g., `data/cart_pole/`, `data/lunar_lander/`). This prevents data from different environments from getting mixed or overwritten, maintaining the self-containment principle.
- Configuration parameters (hyperparameters, environment settings) are usually defined at the beginning of the script or in a dedicated configuration section/object within the solution script.

**Key Technical Decisions (Project-Level):**
- **Language:** Python is the primary language.
- **Core Library:** OpenAI Gymnasium is the foundation for all environments.
- **Virtual Environment:** Use of `GymnasiumVENV/` for managing dependencies.
- **Memory Management:** Cline's Memory Bank system is adopted for context persistence across sessions, with a hierarchical file structure.

**Component Relationships:**
- Each environment's code directory (e.g., `lunar_lander/`) is independent of other environment code directories.
- Solution scripts within an environment directory (e.g., `lunar_lander/lunar_lander.py`) will import utilities from the same directory (e.g., `from .visualization_utils import *` or `from visualization_utils import *`).
- The Memory Bank files are interconnected, with top-level files providing general project context and environment-specific files in `memory-bank/environments/<env_name>/` providing detailed context for that environment.
