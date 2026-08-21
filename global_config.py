from pathlib import Path 

# need to able to find the root of the project 
def find_project_root(marker_files=("global_config.py", ".gitignore")) -> Path:
    current_path = Path(__file__).resolve().parent
    for parent in [current_path, *current_path.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("Project root not found")


# root dir of project
ROOT_DIR = find_project_root()

LAYERED_MDP = "layered_mdp"
UAV = "uav"