"""
Configuration Loader
Loads settings from YAML files and provides easy access to configuration
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """
    Loads and manages configuration from YAML files
    
    Example usage:
        config = ConfigLoader("config/config.yaml")
        model_params = config.get("models.xgboost")
        cost_ratio = config.get("models.cost_ratio", default=400)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config file at this location."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Configuration loaded from: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML file: {self.config_path}\n"
                f"Error: {str(e)}"
            )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., "models.xgboost.n_estimators")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = ConfigLoader()
            >>> config.get("models.cost_ratio")
            400
            >>> config.get("models.xgboost.n_estimators")
            200
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Top-level section name (e.g., "models", "data")
            
        Returns:
            Dictionary containing section configuration
            
        Example:
            >>> config = ConfigLoader()
            >>> config.get_section("models")
            {'cost_ratio': 400, 'xgboost': {...}, 'lightgbm': {...}}
        """
        return self.get(section)
    
    def __repr__(self) -> str:
        """String representation of config loader"""
        return f"ConfigLoader(path={self.config_path})"


# Convenience function for quick loading
def load_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
        
    Example:
        >>> config = load_config()
        >>> cost_ratio = config.get("models.cost_ratio")
    """
    return ConfigLoader(config_path)