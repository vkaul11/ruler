#!/bin/bash
# Navigate to the niah directory
cd data/niah

# Execute the Python script
python niah.py

# Now navigate to the eval directory to run metrics
cd "../../eval"

# Execute the metrics generation script
python generate.py task=niah
