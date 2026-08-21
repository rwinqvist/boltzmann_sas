import os
import joblib
from global_config import ROOT_DIR
from domains.uav.uav_domain import UAVDomain

def get_layouts_path(size, p_obs):
    return f"{ROOT_DIR}/domains/uav/layouts/size{size}_pobs{p_obs}.joblib"


def get_layouts(size, p_obs, num_layouts):
    fn = get_layouts_path(size, p_obs)
    layouts = get_or_create_layouts(size, p_obs, num_layouts, fn)
    return layouts[:num_layouts]


def get_or_create_layouts(size, p_obs, num_layouts: int, fn: str):
    if os.path.exists(fn):
        layouts = joblib.load(fn)
    else: 
        layouts = []

    while len(layouts) < num_layouts: 
        layout = UAVDomain.generate_random_grid(
            size, 
            p_obs
        )

        if layout not in layouts: 
            layouts.append(layout)

    joblib.dump(layouts, fn, compress=3)
    return layouts