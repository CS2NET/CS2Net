# CS<sup>2</sup>Net

Code and dataset for the paper "CS<sup>2</sup>NET: A Causality-Driven Scarcity and Service-Aware Network for Hotel Inventory Management"

#  Due to company policy restrictions, this component will be released upon paper acceptance.

# Overview:

CS<sup>2</sup>Net, a causality-driven framework for intelligent inventory management at Fliggy. CS<sup>2</sup>Net tackles three key challenges in optimizing inventory addition decision: 
 - (1) real-time inventory prediction, by learning a Room Type Scarcity Representation module to infer a hotel’s true availability using internal demand signals and external inventory comparison;
 - (2) reservation acceptance prediction, by leveraging a Hotel Service Representation module to model hotel acceptance probability and reduce the risk of rejected bookings;
 - (3) platform-wide sales uplift, by designing a causal estimation framework that quantifies the Individual Treatment Effect of adding inventory, ensuring invention additions lead to true platform-wide growth rather than simple demand redistribution.


![image](https://github.com/user-attachments/assets/22ba226c-8a43-46fb-b1bc-7e3451a22d37)


- for real-time inventory prediction, we introduce a Room Type Scarcity Representation (RSR) module, which infers the true availability of a room type based on both internal demand
and external inventory comparisons. The Internal Demand-Based Scarcity (IRS) sub-module models the relationship between demand and room availability (e.g., higher demand typically suggests lower availability), while the External Inventory-Based Scarcity (ERS) module estimates room scarcity by analyzing the inventory status of similar hotel-rooms in the same area (e.g., a shortage in the inventory of similar hotels often signals a likely scarcity at the target hotel).
- we design a Hotel Service Representation (HSR) module to obtain hotel-specific service representations, by constructing a hotel-room type service graph that carries key service-related attributes (e.g., historical rejection rates), and enhancing representation learning using graph contrastive learning (GCL).
- to avoid sales shifts and ensure that inventory additions lead to platform-wide sales uplift, we employ a causal inference framework to estimate the Individual Treatment Effect (ITE) of adding inventory. This framework is powered by: Multi-Constraint Disentangled Representation (MDR) module, which mitigates selection bias through adversarial and disentangled learning. Multi-Task Prediction (MTP) module, which jointly models the likelihood of successful reservations and hotel acceptance to compute ITE accurately.


# Environment Settings

Install the Python dependencies:
 
 ```pip install -r requirements.txt```

Install the CUDA version of DGL:

- Download the DGL-Cuda package from [here](https://data.dgl.ai/wheels/cu118/repo.html).
- Install the DGL-Cuda package (e.g., dgl-1.1.2+cu118-cp311-cp311-win_amd64.whl) using pip install dgl-1.1.2+cu118-cp311-cp311-win_amd64.whl.


# Codes & Datasets

To train CS<sup>2</sup>Net:

 ```python train.py```

## Datasets

The CS<sup>2</sup>NET dataset is a large-scale dataset collected in one month in the hotel inventory adjustment scenario of Fliggy, and in order to prevent data crossing problems, the dataset is split into training, validation, and testing sets in chronological order.

 ```train.csv validation.csv test.csv ```


