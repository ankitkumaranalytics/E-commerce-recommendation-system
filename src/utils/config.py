"""
Configuration Loading Module

Loads and manages application configuration from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigLoader:
    """Load and manage application configuration."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Examples:
            config.get("database.type")
            config.get("recommendation.weights.collaborative")
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_all(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Configuration section name
        
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def substitute_env(self, value: Any) -> Any:
        """
        Substitute environment variables in config values.
        
        Supports format: ${VAR_NAME:default_value}
        
        Args:
            value: Value that may contain env variable references
        
        Returns:
            Value with environment variables substituted
        """
        if isinstance(value, str):
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.getenv(var_name, default)
            
            return re.sub(pattern, replace_var, value)
        
        return value

# Global configuration instance
config = ConfigLoader()
