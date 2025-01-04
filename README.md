# MBLP Python Implementation

This repository contains a Python implementation of the mixed binary linear program (MBLP) described in the article [Mathematical Optimization for Real-World Problems](https://www.sciencedirect.com/science/article/pii/S0377221722003071) by (authors' names).

## Datasets

The project utilizes a variety of datasets to evaluate the performance and scalability of the optimization model. The datasets range from small to medium and include a specialized test set with a limited number of customers. Details of the datasets are as follows:

| ID  | Customers | Activities | Eligibility Fraction [%] | Eligibility Patterns |
|-----|-----------|------------|--------------------------|----------------------|
| GS1 | 10,000    | 50         | small (5)                | few (50)             |
| GS5 | 20,000    | 75         | small (5)                | few (50)             |
| GM1 | 100,000   | 100        | small (5)                | few (300)            |
| GM5 | 200,000   | 125        | small (5)                | few (300)            |


