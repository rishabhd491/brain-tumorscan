# This file makes the app directory a proper Python package
# It can be empty or contain initialization code

# For relative imports to work correctly
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 