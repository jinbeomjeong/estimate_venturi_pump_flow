#!/bin/bash

source /home/pi/miniconda3/bin/activate

conda activate tensorflow_220_cpu_python_310

python /home/pi/workspace/estimate_venturi_pump_flow/inference_modbus.py