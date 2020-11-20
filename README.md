# order-inference
Algorithms to infer and evaluate an order in variables, based on the Master thesis of Silvan de Boer that can be found in the file `Order-Based Causal Analysis of Gene Expression Data - Silvan de Boer.pdf`. These algorithms were applied to a gene perturbation dataset by Kemmeren et al., which is typified by high-dimensionality, and a subdivision in observational and interventional data. Interventional data points have a known target variable, and each target variable has only a single intervention data point.

Examples of how the algorithms can be applied can be found in `exp_order_es.py` for the Evolution Strategy methods, and `exp_order.py` for the other methods.

```P. Kemmeren, K. Sameith, L. A. L. van de Pasch, J. J. Benschop, T. L. Lenstra, T. Margaritis, E. O’Duibhir, E. Apweiler, S. van Wageningen, C. W. Ko, and Others. Large-scale genetic perturbations reveal regulatory networks and an abundance of gene-specific repressors. Cell, 157(3):740–752, 2014.```
