# Write Your Abstracts Carefully - The Impact of Abstract Reporting Quality on Findability by Semi-Automated Title-Abstract Screening Tools

### Description
This repository contains the code of a simulation study on the effects of textual characteristics of abstracts on semi-automated title-abstract screening or Technology-Assisted Reviewing (TAR) for evidence synthesis, specifically for systematic reviews. We evaluated the impact of textual quality and other characteristics of abstracts in terms of:

**(I)** Abstract reporting quality, according to the *Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis* (TRIPOD) statement by [Collins *et al.* (2015)](https://pubmed.ncbi.nlm.nih.gov/25623047/), on which more can be found at [www.tripod-statement.org](https://www.tripod-statement.org/).

**(II)** Abstract structural components (length, average sentence length, and structured vs unstructured)

**(III)** Abstract terminology usage, computed by the deviation in TF-IDF vector values as a measure of aberrant term occurrence

### Python codes
- compute_rankings.py
- data_preprocessing.ipynb
- set-up.ipynb
  
### Data
The reviews (Table 1) are listed and the corresponding PubMed identifiers (pmids) of the studies including the full-text labels are available in the data folder.

##### Table 1 | The datasets of previously conducted systematic reviews that were used in the simulation study

| Review ID | Review number | Total records    | Relevant records (%)    | Final inclusions | Reference | Title |
| --- | :---:   | :---: | :---: | :---: | :---: | :---: |
| Prog1_reporting | 1 | 2482   | 312 (12.6)   | 152 | Andaur Navarro *et al.* (2022) | Completeness of reporting of clinical prediction models developed using supervised machine learning: a systematic review|
| Prog3_tripod | 3 | 4871   | 347 (7.1)   | 147 | Heus *et al.* (2018) | Poor reporting of multivariable prediction model studies: towards a targeted implementation strategy of the TRIPOD statement|

