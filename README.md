# X-MedRELA
Source code for ACL 2020 paper "[Rationalizing Medical Relation Prediction from Corpus-level Statistics](https://zhenwang9102.github.io/pdf/ACL2020_ZW_X_MedRELA.pdf)".

## Introduction


<p align="center">
<img src="toy_example.png" alt="a toy example" width="500" title="A Toy Example"/>
</p>

In this project, we propose an interpretable framework to rationalize medical relation prediction based on corpus-level statistics. An toy example to illustrate our intuition is shown above.

<p align="center">
<img src="framwork_workflow.png" alt="workflow" width="550" title="Framework Workflow"/>
</p>

Our framework is inspired by existing cognitive theories on human memory recall and recognition, and an be easily understood by users as well as provide reasonable explanations to justify its prediction. Its workflow is shown above. Essentially, it leverages corpus-level statistics to recall associative contexts and recognizes their relational connections as model rationales.

## Dataset

## Run
To train the model, simply run the following scrips:

  > bash ./src/bash.sh

## Citation
```
@inproceedings{wang2020rationalizing,
  title={Rationalizing Medical Relation Prediction from Corpus-level Statistics},
  author={Wang, Zhen and Lee, Jennifer and Lin, Simon and Sun, Huan},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
