# Write Your Abstracts Carefully - The Impact of Abstract Reporting Quality on Findability by Semi-Automated Title-Abstract Screening Tools
<br>

This repository corresponding to the study on the impact of abstract reporting quality on findability by semi-automated screening tools: <br>

**I. Spiero, A.M. Leeuwenberg, K.G.M. Moons, L. Hooft, J.A.A. Damen (2025). Write Your Abstracts Carefully - The Impact of Abstract Reporting Quality on Findability by Semi-Automated Title-Abstract Screening Tools. *Submitted*** 

### Description
This repository contains the code of a simulation study on the effects of textual characteristics of abstracts on semi-automated title-abstract screening or Technology-Assisted Reviewing (TAR) for evidence synthesis, specifically for systematic reviews. We evaluated the impact of textual quality and other characteristics of abstracts in terms of:

**(I)** Abstract reporting quality, according to the *Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis* (TRIPOD) statement by [Collins *et al.* (2015)](https://pubmed.ncbi.nlm.nih.gov/25623047/), on which more can be found at [www.tripod-statement.org](https://www.tripod-statement.org/).

**(II)** Abstract structural components

**(III)** Abstract terminology usage

### Codes
- ```data_preprocessing.ipynb``` : to clean the raw TRIPOD scores and add the other abstracts characterstics for the available systematic review datasets
- ```compute_rankings.py``` : to compute rankings when using a semi-automated title-abstract screening tool
- ```results_generation.ipynb``` : to use the abstract data and rankings to assess the association between the abstract characterstics and ranking positions
  
### Data
The systematic reviews (Table 1) are listed and their corresponding data are available upon request.

##### Table 1 | The datasets of previously conducted systematic reviews that were used in the simulation study

| Review ID | Review number | Total records    | Relevant records (%)    | Final inclusions | Reference | Title |
| --- | :---:   | :---: | :---: | :---: | :---: | :---: |
| Prog1_reporting | 1 | 2482   | 312 (12.6)   | 152 | Andaur Navarro *et al.* (2022) | [Completeness of reporting of clinical prediction models developed using supervised machine learning: a systematic review](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01469-6)|
| Prog3_tripod | 2 | 4871   | 347 (7.1)   | 147 | Heus *et al.* (2018) | [Poor reporting of multivariable prediction model studies: towards a targeted implementation strategy of the TRIPOD statement](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-018-1099-2)|

