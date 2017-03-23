#!/bin/bash

rowct=17000

## must be run from this location
head -n $rowct data/brand_finger_training_multiple_cats.csv | cut -d ',' -f 2,3,306-308 --complement > data/sampledata/sampledata_qty_amt.csv
head -n $rowct data/brand_finger_training_multiple_cats.csv | cut -d ',' -f 2,3,305,307,308 --complement > data/sampledata/sampledata_rev_amt.csv
head -n $rowct data/brand_finger_training_multiple_cats.csv | cut -d ',' -f 2,3,305,306,308 --complement > data/sampledata/sampledata_qty_share.csv
head -n $rowct data/brand_finger_training_multiple_cats.csv | cut -d ',' -f 2,3,305-307 --complement > data/sampledata/sampledata_rev_share.csv

