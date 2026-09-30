import yaml

def load_subsystems(config_path="config/subsystems.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_subsystem(feature_name, mapping_config):
    mapping = mapping_config.get("subsystem_mapping", {})
    default = mapping_config.get("default_subsystem", "Unknown / Unmapped")
    return mapping.get(feature_name, default)
