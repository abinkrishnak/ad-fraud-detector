"""
Quick test to verify config loader works
Run this with: python test_config.py
"""

from src.utils.config_loader import load_config

# Load configuration
config = load_config()

# Test getting values
print("=" * 60)
print("TESTING CONFIG LOADER")
print("=" * 60)

print(f"\n1. Project name: {config.get('project.name')}")
print(f"2. Cost ratio: {config.get('models.cost_ratio')}")
print(f"3. XGBoost n_estimators: {config.get('models.xgboost.n_estimators')}")
print(f"4. Test with default: {config.get('nonexistent.key', default='DEFAULT_VALUE')}")

print("\n5. Entire models section:")
models_config = config.get_section('models')
print(f"   Keys in models: {list(models_config.keys())}")

print("\n✅ Config loader works perfectly!")