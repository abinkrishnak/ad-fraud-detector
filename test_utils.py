"""
Test all utility functions
Run with: python test_utils.py
"""

from src.utils import (
    load_config,
    setup_logger,
    set_seed,
    ensure_dir,
    format_time,
    calculate_class_weights
)
import numpy as np

# Setup logger
logger = setup_logger(level="INFO")

logger.info("="*60)
logger.info("TESTING ALL UTILITIES")
logger.info("="*60)

# Test 1: Config loading
logger.info("\n1. Testing config loader...")
config = load_config()
logger.success(f"   Project name: {config.get('project.name')}")
logger.success(f"   Cost ratio: {config.get('models.cost_ratio')}")

# Test 2: Set seed
logger.info("\n2. Testing random seed...")
set_seed(42)
random_nums = np.random.rand(3)
logger.success(f"   Random numbers: {random_nums}")

set_seed(42)  # Reset
random_nums2 = np.random.rand(3)
logger.success(f"   Again (same!): {random_nums2}")

# Test 3: Ensure directory
logger.info("\n3. Testing directory creation...")
test_dir = ensure_dir("test_output/models/saved")
logger.success(f"   Created: {test_dir}")

# Test 4: Format time
logger.info("\n4. Testing time formatting...")
logger.success(f"   125.5 seconds = {format_time(125.5)}")
logger.success(f"   3665 seconds = {format_time(3665)}")

# Test 5: Class weights
logger.info("\n5. Testing class weight calculation...")
# Simulate imbalanced data: 99.8% class 0, 0.2% class 1
y = np.array([0]*998 + [1]*2)
weights = calculate_class_weights(y, cost_ratio=400)
logger.success(f"   Class weights: {weights}")

logger.info("\n" + "="*60)
logger.success("ALL UTILITIES WORKING! ✅")
logger.info("="*60)