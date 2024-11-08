#!/bin/bash
# Navigate to the hashhop directory
cd data/hashhop

# Execute the Python script to generate the dataset
#python hashhop.py

# Now navigate to the eval directory to run metrics
cd ../../eval

# Execute the metrics generation script
python generate.py task=hashhop

