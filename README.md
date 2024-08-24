# predictability_tipping
## Folders
```
.
├── tipping_element
│   ├── amazon_tip_num_obs.py
│   └── amazon_tip_num_non_obs.py
│   ├── AMOC_tip_num_obs.py
│   └── AMOC_tip_num_non_obs.py
│   ├── amazon.py
│   └── AMOC.py
│   ├── figure2_heatmap.py
│   └── figure3_heatmap.py
│   ├── figure2.py
│   ├── figure3.py
│   ├── figure4.py
│   ├── temp_scenario.py
├── particle_filter
│   ├── particle.py
│   └── __init__.py
├── data
|    ├── final_result : In this folder, the results of experiment1 and 2 are included. This results are made by amazon_tip_num_obs.py, amazon_tip_num_non_obs.py,amoc_tip_num__obs.py and amoc_tip_num_non_obs.py. The figures of the results are in /output/amazon or /output/amoc. 
|       └── amazon
|       └── amoc
├── output
|   |── amazon
|   |    └── example : Results of amazon.py to show the examples of data assimilation.
|   |    └── rms : The 30-year average internal variability data and figures. Tempearture scenarios are also in this directory.
    ├── amoc : The same to ../amazon
    └── final_result
         └── amazon
         ├── amoc
```
## TO DO
Before runnning, you have to change data directory and figure directory in `.sh` files. 

In order to gain the results of experiment1 & figure2, run
`./experiment1.sh`

In order to gain the results of experiment2 & figure3, 4, run
`./experiment2.sh`
