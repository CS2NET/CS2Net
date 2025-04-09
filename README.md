# CS<sup>2</sup>Net

Code and dataset for the paper "CS<sup>2</sup>NET: A Causality-Driven Scarcity and Service-Aware Network for Hotel Inventory Management"

The CS<sup>2</sup>NET dataset is a large-scale dataset collected in one month in the hotel inventory adjustment scenario of Fliggy.

# Overview:

CS<sup>2</sup>Net, a causality-driven framework for intelligent inventory management at Fliggy. CS<sup>2</sup>Net tackles three key challenges in optimizing inventory addition decision: 
 - (1) real-time inventory prediction, by learning a Room Type Scarcity Representation module to infer a hotel’s true availability using internal demand signals and external inventory comparison;
 - (2) reservation acceptance prediction, by leveraging a Hotel Service Representation module to model hotel acceptance probability and reduce the risk of rejected bookings;
 - and (3) platform-wide sales uplift, by designing a causal estimation framework that quantifies the Individual Treatment Effect of adding inventory, ensuring invention additions lead to true platform-wide growth rather than simple demand redistribution.

Extensive offline experiments and a two-week online A/B test on Fliggy demonstrate CS<sup>2</sup>Net’s superiority over state-of-the-art baselines. CS2Net is now deployed in Fliggy’s Smart Inventory Management system, supporting real-time hotel inventory management.

And in order to prevent data crossing problems, the dataset is split into training, validation, and testing sets in chronological order.


![image](https://github.com/user-attachments/assets/22ba226c-8a43-46fb-b1bc-7e3451a22d37)
