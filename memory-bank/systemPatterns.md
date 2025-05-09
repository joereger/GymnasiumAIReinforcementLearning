# System Patterns: Gymnasium Environment Solutions

**Overall Repository Structure:**
- The root directory contains Python scripts, each typically representing a solution or experiment for a specific Gymnasium environment (e.g., `bipedal_walker.py`, `cart_pole.py`).
- A `data/` directory may be used for storing experiment results, trained models, or other artifacts.
- A `GymnasiumVENV/` directory houses the Python virtual environment for the project, ensuring consistent dependencies.
- A `visualization_utils.py` script provides common utilities for visualizing environment states or agent performance.
- The `memory-bank/` directory contains all documentation for Cline's Memory Bank.

**Memory Bank Structure:**
- **Top-Level Files:** Located directly in `memory-bank/`, these files provide overall project context (brief, product context, general system patterns, overall tech context, active project-level work, and overall progress).
- **`memory-bank/environments/` Directory:** This directory contains subfolders for each specific Gymnasium environment being worked on.
    - **`memory-bank/environments/<environment_name>/`:** Each subfolder holds a set of Memory Bank files (`environment_brief.md`, `approaches.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`) dedicated to that particular environment, ensuring isolated and detailed context.

**Code Organization within Environment Solutions:**
- Typically, each Python script (e.g., `bipedal_walker_a3c.py`) will contain the full implementation for a specific agent solving a specific environment.
- Common functionalities (like visualization) are refactored into shared utility scripts (e.g., `visualization_utils.py`).
- Configuration parameters (hyperparameters, environment settings) are usually defined at the beginning of the script or in a dedicated configuration section/object.

**Key Technical Decisions (Project-Level):**
- **Language:** Python is the primary language.
- **Core Library:** OpenAI Gymnasium is the foundation for all environments.
- **Virtual Environment:** Use of `GymnasiumVENV/` for managing dependencies.
- **Memory Management:** Cline's Memory Bank system is adopted for context persistence across sessions, with a hierarchical file structure.

**Component Relationships:**
- Individual environment solution scripts (e.g., `lunar_lander.py`) are largely independent but may use `visualization_utils.py`.
- The Memory Bank files are interconnected, with top-level files providing general context and environment-specific files providing detailed context.
