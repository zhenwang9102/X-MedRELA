# X-MedRELA
Source code for ACL 2020 paper "[Rationalizing Medical Relation Prediction from Corpus-level Statistics](https://zhenwang9102.github.io/pdf/ACL2020_ZW_X_MedRELA.pdf)".

## Introduction

Generating explanations for the decision made by advanced machine learning models, such as neural networks, has drawn extensive attention recently, and is especially important in those high-stake domains such as medicine, finance, and the judiciary. Reasonable explanations could help debug themodel, detect model bias, and more importantly, earn user trust for practical applications.

<p align="center">
<img src="toy_example.png" alt="a toy example" width="500" title="A Toy Example"/>
</p>

In this paper, we propose to explain medical relation prediction based on existing cognitive theories about human memory recall and recognition. Our intuition is that, to explain the relationships between two entities, we humans tend to resort to the connections between their contexts. An toy example to illustrate our intuition is shown above. For example, to predict why “Aspirin” may treat “Headache”, a model could first *recall* a relevant entity “PainRelief” for “Headache” as they co-occur frequently, and then *recognize* there is a chance that“Aspirin” can lead to “Pain Relief”, based on which it could finally make a correct prediction ( Aspirinmay treat Headache).

<p align="center">
<img src="framwork_workflow.png" alt="workflow" width="550" title="Framework Workflow"/>
</p>

Inspired by such cognitive processes, we build a graph-based framework to rationalize medical relationprediction based on corpus-level statistics. The task is to predict the relations between two medicalterms. Its workflow is shown above. The framework consists of three cognitive stages: association recall, assumption recognition,and decision making, which can be easily understood by end users and generate reasonable rationalesto justify the model prediction. Essentially, our framework leverages corpus-level statistics to recallassociative contexts of target entities and recognizes their relational connections as model rationales. We show its competitive predictive performance compared with a comprehensive list of black-box neural models and demonstrate the quality of model rationales via expert evaluations.


## Dataset
You can download the data from the following link: [Corpus-level Statistics](https://drive.google.com/file/d/1nwVPdxP1p7NkrD6N3isSGTL2iJtv9r8u/view?usp=sharing), [Labeled Relation Data](https://drive.google.com/file/d/1iqT8oswl3E9-c8Iirv7UAKD5GgQIhlT8/view?usp=sharing), [Relation List](https://drive.google.com/file/d/10ijyAY0OXCCVEXP4n5clRpMpKMc6eMCb/view?usp=sharing), [Relation Triples](https://drive.google.com/file/d/1TXVcAzzH7fq1kAH7B3PWfeWwgRhh_c_b/view?usp=sharing).

## Run
To train the model, simply run the following scrips:
```
> bash ./src/bash.sh
```

After the traning, you can infer the rationales that are important for the model prediction:
```
> bash ./src/bash_infer.sh
```

If you have any questions, please feel free to contact us! Also, feel free to check other tools in our group (https://github.com/sunlab-osu) 😊


## Citation
```
@inproceedings{wang2020rationalizing,
  title={Rationalizing Medical Relation Prediction from Corpus-level Statistics},
  author={Wang, Zhen and Lee, Jennifer and Lin, Simon and Sun, Huan},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
