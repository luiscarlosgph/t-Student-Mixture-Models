#!/bin/bash
#
# @brief  Run tests for the optimization of t-Student mixtures.  
#
# @author Luis Carlos Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
# @date   11 Jul 2017.

# Folders
TEMP_TEST_FOLDER='temp_test_dir'
TEST_FOLDER='tests'
SRC_FOLDER='src'

# Script files
PYTHON_SCRIPT='smm.py'
TEST_SCRIPT='smm_test.py'

# Create temporary test folder
mkdir $TEMP_TEST_FOLDER

# Copy files to the temporary folder
cp "$SRC_FOLDER/$PYTHON_SCRIPT" $TEMP_TEST_FOLER/
cp "$TEST_FOLDER/$TEST_SCRIPT"  $TEMP_TEST_FOLER/

# Run tests
python "$TEMP_TEST_FOLDER/$TEST_SCRIPT" 

# Delete temporary folder
rm -r "$TEMP_TEST_FOLDER"
