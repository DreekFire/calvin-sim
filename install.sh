#!/bin/bash

# Wheel is never depended on, but always needed. MulticoreTSNE requires lower CMake version
python -m pip install wheel cmake==3.18.4

cd calvin_env/tacto
python -m pip install -e .
cd ..
python -m pip install -e .
cd ../calvin_models
python -m pip install -e .
