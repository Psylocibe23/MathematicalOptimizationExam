# MBLP Python Implementation

This repository contains a Python implementation of the mixed binary linear program (MBLP) described in the article [Mathematical Optimization for Real-World Problems](https://www.sciencedirect.com/science/article/pii/S0377221722003071) by T. Bigler, M. Kammermann, P. Baumann.

## Datasets

The datasets range from small to medium and include a specialized test set with a limited number of customers (not included in the scalability analysis). Details of the datasets are as follows:

| ID  | Customers | Activities | Eligibility Fraction [%] | Eligibility Patterns |
|-----|-----------|------------|--------------------------|----------------------|
| GS1 | 10,000    | 50         | small (5%)               | few (50)             |
| GS5 | 20,000    | 75         | small (5%)               | few (50)             |
| GM1 | 100,000   | 100        | small (5%)               | few (300)            |
| GM5 | 200,000   | 125        | small (5%)               | few (300)            |

## Implementation Files

- **MBLP_class.py**: Contains the common constraints and variables used across all MBLP models detailed in the article.
- **MBLP.py**: Inherits from MBLP_class and implements the first MBLP model described in the article. It excludes the minimum contact constraints due to the absence of explicit rules for minimum contact in the source article.
- **matheuristic.py** and **alternative_MBLP.py**: These files are placeholders for now and contain no implementation.
- **scalability_analysis.py**: Conducts scalability analysis on the datasets GS1, GS5, GM1, and GM5. The complete results of the analysis are documented at the bottom of the file as a comment.





