#!/bin/bash
#
# @brief  Check how much of the 'smm' module is covered by the tests.
#
# @author Luis Carlos Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
# @date   11 Jul 2017.

# Folders
TEMP_TEST_FOLDER='temp_test_dir'
TEMP_COVERAGE_FOLDER='coverage'
TEST_FOLDER='tests'
SRC_FOLDER='src'

# Script files
PYTHON_SCRIPT='smm.py'
TEST_SCRIPT='smm_test.py'

# Create temporary test folder
mkdir $TEMP_TEST_FOLDER

# Copy files to the temporary folder
cp "$SRC_FOLDER/$PYTHON_SCRIPT" "$TEMP_TEST_FOLDER/$TEMP_COVERAGE_FOLDER"
cp "$TEST_FOLDER/$TEST_SCRIPT"  "$TEMP_TEST_FOLDER/"

# Run tests
python "$TEMP_TEST_FOLDER/$TEST_SCRIPT" 

# Delete temporary folder
rm -r "$TEMP_TEST_FOLDER"
