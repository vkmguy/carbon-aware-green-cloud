# A Simulation Based carbon-aware approach to reduce CO2 emissions in Cloud


This repository contains datasets, simulation code, and analysis notebooks used to calculate the carbon emissions for ad-hoc workloads shifted towards the weekends:

- `data/*`: Energy production and carbon intensity datasets for the regions Germany, Great Britain, France (all via the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)) and California (via [California ISO](https://www.caiso.com/)) for the entire year 2020 +-10 days.
- `compute_carbon_intensity.py`: The script used to convert energy production to carbon intensity data using energy source carbon intensity values provided by an [IPCC study](http://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf).
- `simulate.py`: A simulator to experimentally evaluate temporal workload shifting approaches in data centers with the goal to consume low-carbon energy.
- `analysis.ipynb`: Notebook used to analyze the carbon intensity data.
- `evaluation.ipynb`: Notebook used to analyze the simulation results.

For executing the code you need to install the libraries listed in `environment.yml`, e.g. by using a [conda environment](https://conda.io/).

Steps to run the experiments:
- Execute the main method of `compute_carbon_intensity.py`
- Above steps generates carbon intensity data which can be used for analyis and geenrating results
- Execute `analysis.ipynb` for generating analysis of data
- Execute `simulate.py` for running the simulations
- Execute `evaluation.ipynb` for analyzing the generated results

This study is the extension of the study from presented by the authors of the paper "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud" to include the shifting potential of the ad-hoc workloads to the weekends by adding a scheduling algorithm for the weekends which is added in the  file `strategy.py`. 

## Reference
A great thanks to the authors of "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud" for providing this amazing framework and making this study possible.

- Philipp Wiesner, Ilja Behnke, Dominik Scheinert,  Kordian Gontarska, and Lauritz Thamsen. "[Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud](https://arxiv.org/pdf/2110.13234.pdf)" In the Proceedings of the *22nd International Middleware Conference*, ACM, 2021.

BibTeX:
```
@inproceedings{Wiesner_LetsWaitAwhile_2021,
  author={Wiesner, Philipp and Behnke, Ilja and Scheinert, Dominik and Gontarska, Kordian and Thamsen, Lauritz},
  booktitle={Middleware'21: 22nd International Middleware Conference}, 
  title={Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud}, 
  publisher = {{ACM}},
  year={2021},
  doi={10.1145/3464298.3493399}
}
```
