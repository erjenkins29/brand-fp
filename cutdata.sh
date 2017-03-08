#!/bin/bash

## must be run from this location
head data/brand_finger_training_nutrition.csv | cut -d ',' -f 2,3,306 --complement > sampledata.csv
