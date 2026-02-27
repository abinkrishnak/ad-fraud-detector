"""
Test logger functionality
Run with: python test_logger.py
"""

from src.utils.logger import setup_logger

# Setup logger (will save to logs/test.log)
logger = setup_logger(level="DEBUG", log_file="logs/test.log")

print("=" * 60)
print("TESTING LOGGER")
print("=" * 60)
print()

# Test different log levels
logger.debug("This is a DEBUG message (very detailed)")
logger.info("This is an INFO message (normal operation)")
logger.warning("This is a WARNING message (something unusual)")
logger.error("This is an ERROR message (something failed)")
logger.success("This is a SUCCESS message (something worked!)")

print()

# Test logging with variables
model_name = "XGBoost"
accuracy = 0.9712
logger.info(f"Model {model_name} achieved accuracy: {accuracy:.2%}")

# Test logging a fake error
try:
    result = 10 / 0  # This will cause error
except ZeroDivisionError as e:
    logger.error(f"Caught an error: {e}")
    logger.exception("Full traceback:")  # Shows detailed error info

print()
logger.info("✅ Logger test complete! Check 'logs/test.log' file")