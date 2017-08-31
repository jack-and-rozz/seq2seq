#!/bin/bash

select-gpu.sh 0 nohup ./run.sh bins/graph.py m_graph/wn.ModelE train configs/graph/config --model_type=ModelE --output_log=True& 
select-gpu.sh 1 nohup ./run.sh bins/graph.py m_graph/wn.DistMult train configs/graph/config --model_type=DistMult --output_log=True&

