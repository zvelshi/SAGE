# default
import os
import datetime
import shutil
import pandas as pd
import yaml

# ours
from utils.logging_setup import get_logger

log = get_logger(__name__)


def new_run_dir(mode: str) -> str:
    """Create a timestamped ``out/<mode>/<ts>/`` directory and return its path."""
    run_dir = os.path.join("out", mode, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_configs(run_dir, config_files, hardpoints_name):
    """
    Copies relevant config files to the run directory for reproducibility.
    """
    for cfg in config_files:
        if os.path.exists(cfg):
            shutil.copy(cfg, run_dir)
            log.debug("backed up config %s", cfg)

    hp_path = f"config/hardpoints/{hardpoints_name}.yml"
    if os.path.exists(hp_path):
        shutil.copy(hp_path, run_dir)
        log.debug("backed up hardpoints %s", hp_path)
    else:
        log.warning("could not find hardpoints file to back up: %s", hp_path)

def pack_points_nicely(vehicle, id, step):
    """
    Take the step output, combine it with hardpoint names and fixed hardpoints for one dictionary
    """
    pkg = {}
    hp = vehicle.get_corner_from_id(id).hardpoints
    
    for name in hp.names:
        if name in step.keys():
            pkg[name] = step[name]
        else:
            pkg[name] = getattr(hp, name)
            
    return pkg

def export_static_hardpoints(vehicle, settled_step, hardpoints_name, run_dir):
    """
    Copy the vehicle's source hardpoints YAML, inject the settled (world-frame)
    inboard/outboard points and CoG height from a StaticDrop run, and save it as
    '<hardpoints_name>_NEW_STATIC.yml' in the run directory.
    """
    src_path = f"config/hardpoints/{hardpoints_name}.yml"
    if not os.path.exists(src_path):
        log.warning("could not find hardpoints file to export: %s", src_path)
        return None

    with open(src_path) as f:
        data = yaml.safe_load(f)

    nickname = list(data.keys())[0]
    root = data[nickname]

    root["mass_properties"]["cog"] = [
        round(float(v), 3) for v in settled_step["cog_pos"]
    ]

    for half, step_key in (("front", "fr"), ("rear", "rr")):
        hp_cls = type(vehicle.get_corner_from_id([1, 0 if half == "front" else 1]).hardpoints)
        root[half].update(hp_cls.points_to_yaml(settled_step[step_key]))

    export_path = os.path.join(run_dir, f"{hardpoints_name}_NEW_STATIC.yml")
    with open(export_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    log.info("exported settled hardpoints to %s", export_path)
    return export_path

def export_extreme_points_to_xlsx(results, run_dir, sweep, template_path="example.xlsx"):
    log.debug("generating extreme-points xlsx export")
    
    if not os.path.exists(template_path):
        log.warning("xlsx template not found, skipping export: %s", template_path)
        return

    df_template = pd.read_excel(template_path, header=None)
    header_row_0 = df_template.iloc[0].fillna("").tolist() 
    header_row_1 = df_template.iloc[1].fillna("").tolist() 

    steer_min = sweep.steer.min
    steer_max = sweep.steer.max
    
    row_targets = [
        ("Static",         "static", "0_steer"),
        ("Droop",          "droop",  "0_steer"),
        ("Jounce",         "jounce", "0_steer"),
        ("Static-Steer_L", "static", f"{steer_max}_steer"),
        ("Droop-Steer_L",  "droop",  f"{steer_max}_steer"),
        ("Jounce-Steer_L", "jounce", f"{steer_max}_steer"),
        ("Static-Steer_R", "static", f"{steer_min}_steer"),
        ("Droop-Steer_R",  "droop",  f"{steer_min}_steer"),
        ("Jounce-Steer_R", "jounce", f"{steer_min}_steer")
    ]
    
    point_mapping = {
        'Lower_Wishbone_Front_Pivot':      ('front', 'lf'),
        'Lower_Wishbone_Rear_Pivot':       ('front', 'lr'),
        'Lower_Wishbone_Outer_Ball_Joint': ('front', 'lbj'),
        'Upper_Wishbone_Front_Pivot':      ('front', 'uf'),
        'Upper_Wishbone_Rear_Pivot':       ('front', 'ur'),
        'Upper_Wishbone_Outer_Ball_Joint': ('front', 'ubj'),
        'Damper_Wishbone_End':             ('front', 's_ob'),
        'Damper_Body_End':                 ('front', 's_ib'),
        'Outer_Track_Rod_Ball_Joint':      ('front', 'tr_ob'),
        'Outer_Track_Ball_Joint':          ('front', 'tr_ob'),
        'Inner_Track_Rod_Ball_Joint':      ('front', 'tr_ib'),
        'Wheel_Spindle_Point':             ('front', 'piv_ob'),
        'Wheel_Centre_Point':              ('front', 'wc'),
        'Inboard_CV_Centre':               ('front', 'piv_ib'),
        'Inboard_CV_Axis_Point':           ('front', 'piv_ib'), 
        'Inner_CV_Axis_Point':             ('front', 'piv_ib'),
        
        'Front_Trailing_Link_Pivot':       ('rear', 'tl_f'),
        'Bottom_Inner_Camber_Link_Pivot':  ('rear', 'lcl_ib'),
        'Bottom_Outer_Camber_Link_Pivot':  ('rear', 'lcl_ob'),
        'Top_Inner_Camber_Link_Pivot':     ('rear', 'ucl_ib'),
        'Top_Outer_Camber_Link_Pivot':     ('rear', 'ucl_ob'),
        'Bottom_Shock_Mount':              ('rear', 's_ob'),
        'Top_Shock_Mount':                 ('rear', 's_ib'),
        'Outboard_CV_Center':              ('rear', 'piv_ob'),
        'Wheel_Centre':                    ('rear', 'wc'),
        'Inboard_CV_Center':               ('rear', 'piv_ib')
    }

    hp_name = sweep.hardpoints
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    output_data = [header_row_0, header_row_1]

    for row_name_prefix, cond, steer_front in row_targets:
        row_name = f"{row_name_prefix}_{date_str}"
        
        new_row = [""] * len(header_row_1)
        new_row[0] = row_name
        new_row[1] = "Generated via Simulation"
        
        steer_rear = "0_steer" # Rear never steers
        
        for col_idx, col_header in enumerate(header_row_1):
            if col_idx < 2 or not col_header:
                continue
                
            base_name = col_header.split('@')[0]
            
            is_mirrored = False
            if base_name.startswith("M_"):
                is_mirrored = True
                base_name = base_name[2:]
            
            axis = base_name[-1]      
            clean_name = base_name[:-2] 
            
            if clean_name in point_mapping:
                half, short_name = point_mapping[clean_name]
                
                target_side = 'right' if is_mirrored else 'left'
                steer_val = steer_front if half == 'front' else steer_rear
                source_data = results.get(half, {}).get(target_side, {}).get(cond, {}).get(steer_val, {})
                
                if short_name in source_data:
                    coords = source_data[short_name] 
                    val = 0.0
                    if axis == 'X': val = coords[0]
                    elif axis == 'Y': val = coords[1]
                    elif axis == 'Z': val = coords[2]
                    if is_mirrored and axis == 'Y':
                        val = -val
                        
                    new_row[col_idx] = round(val, 3)

        output_data.append(new_row)

    export_path = os.path.join(run_dir, f"HARDPOINTS_{hp_name}.xlsx")
    df_out = pd.DataFrame(output_data)
    df_out.to_excel(export_path, index=False, header=False)
    
    log.info("extreme-points design table exported to %s", export_path)