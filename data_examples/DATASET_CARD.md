# Dataset Card for ClinicalBench

ClinicalBench: An End-to-End, Real-Case-based, Data-Leakage-Free Benchmark for Multi-Department Clinical Diagnostic Evaluation

## Table of Contents

- [Dataset Description](##dataset-description)
  - [Dataset Summary](###dataset-summary)
  - [Supported Tasks](#supported-tasks)
  - [Languages](###languages)
- [Dataset Structure](#dataset-structure)
  - [Data Fields](#data-fields)
  - [Data Instances](#data-instances)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Data Sources & Licenses](#data-sources--licenses)
  - [Data Annotations & Quality](#data-annotations--quality)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Discussion of Biases](#discussion-of-biases)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** https://github.com/WeixiangYAN/ClinicalLab
- **Repository:** https://github.com/WeixiangYAN/ClinicalLab
- **Paper:** https://arxiv.org/pdf/2406.13890
- **Point of Contact:** Weixiang Yan (yanweixiang.ywx@gmail.com)

### Dataset Summary

ClinicalBench is a fine-grained evaluation benchmark specifically designed for multi-departmental clinical diagnosis, covering 24 departments such as pediatrics, orthopedics, and neurosurgery. It involves 150 different diseases, each comprising 10 specific cases, totaling 1500 samples, with an average of about 1000 tokens per case. Each case in ClinicalBench contains detailed clinical data, such as the patient's gender, age, chief complaint, medical history, and physical examination. Additionally, the cases include various medical imaging reports, such as X-rays, computed tomography (CT) scans, magnetic resonance imaging (MRI), and ultrasound examinations, as well as biochemical, immunological, microbiological, and pathological laboratory examination results from biological samples such as blood, urine, and cerebrospinal fluid.

### Supported Tasks

ClinicalBench systematically evaluates the end-to-end practicality of LLMs in clinical diagnosis by simulating the complete patient visit process, from the patient's entry into the hospital to their discharge. We divide the entire process into 8 specific tasks, covering various stages from preliminary reception to final diagnosis and treatment plan formulation. These tasks are: Department Guide, Preliminary Diagnosis, Diagnostic Basis, Differential Diagnosis, Final Diagnosis, Principle of Treatment, Treatment Plan, Imaging Diagnosis. The detailed task introduction can be found in Section 3.4 of the paper.

### Languages

Chinese and English

## Dataset Structure

The ClinicalBench dataset adopts an application access system. After users agree to the [ClinicalBench Usage and Data Distribution License Agreement](./DATA_LICENSE.pdf) and submit an application, we will send the dataset to the email address provided by the user within 48 hours. You can apply for access to and use of the ClinicalBench dataset at the following URL: [https://forms.gle/Tkq5UTinW7bBB6388](https://forms.gle/Tkq5UTinW7bBB6388)

### Data Fields

Data samples are arranged as follows:

- `id`: the counting sequence number of the data sample.
- `clinical_case_uid`: the identification number of the data sample.
- `language`: the language of the data sample, there are two options: Chinese(zh) and English(en).
- `clinical_department`: the name of the department corresponding to the data sample.
- `principal_diagnosis`: the final diagnosis results annotated by human doctors.
- `preliminary_diagnosis`: the preliminary diagnosis results annotated by human doctors.
- `diagnostic_basis`: the diagnostic basis annotated by human doctors for the preliminary diagnosis results.
- `differential_diagnosis`: the human doctor lists the possible diseases that may be causing the current symptoms of the patient corresponding to the data sample, and briefly explains why they are excluded.
- `treatment_plan`: the treatment plans annotated by human doctors.
- `clinical_case_summary`: the patient's chief complaint, physical examination, medical history, auxiliary examinations and other information.
- `imageological_examination` is a list of the necessary imageological examinations prescribed by human doctors for patients corresponding to the data samples, each imageological examination contains:
  - `findings`: a detailed natural language description written by a human doctor based on the image.
  - `impression`: the diagnosis results written by human doctors based on the findings.
- `laboratory_examination` is a list of the necessary laboratory examinations prescribed by human doctors for patients corresponding to the data samples, each laboratory examination contains:
  - `findings`: the natural language descriptions of laboratory test results.
  - `impression`: the diagnosis results written by human doctors based on the findings.
- `pathological_examination`: the pathological examination results of the patients corresponding to the data samples.
- `therapeutic_principle`: the treatment principles annotated by human doctors.

### Data Instances

Please see [data_example_zh.json](./data_example_zh.json) and [data_example_en.json](./data_example_en.json).

### Data Splits

N/A, this is an evaluation benchmark.

## Dataset Creation

The dataset was created from the middle of 2023 to early 2024 at the Vaneval AI.

### Curation Rationale

Recent studies find that existing benchmarks cannot effectively evaluate the medical capabilities of LLMs. Firstly, existing benchmarks are often based on data collected from online consultation platforms or medical textbooks, which could easily be included in the training data of LLMs, that is, leading to **data leakage or contamination** and thus biasing the performance evaluation of LLMs. Secondly, the departmental setup in modern medicine is designed to address the complex medical needs of different structures and functions of human organs. The specific skills and treatment methods vary significantly across different departments. However, existing evaluation benchmarks overlook the characteristics of **multi-departmental and highly specialized nature** of modern medicine, hence they are insufficient in capturing performance differences across departments. Thirdly, existing evaluation methods typically confine themselves to multiple-choice questions, which does **not align with real-world clinical diagnostic scenarios**. In actual medical environments, patients seek medical services because they are uncertain about their health conditions, rather than knowing the possible disease options and then seeking a doctor's judgment. Last but not least, there is currently no evaluation method that can comprehensively and reliably evaluate the **end-to-end practicality** of LLMs in the entire clinical diagnosis process, which starts from the moment a patient enters the clinic and ends when the patient is discharged. This issue will in turn limit the design and evaluation of practical medical agents powered by LLMs and harm exploitation of the full potential of LLMs.

To address these limitations, we introduce ClinicalBench, an end-to-end multi-departmental clinical diagnostic evaluation benchmark for effectively and comprehensively evaluating the clinical diagnostic capabilities of LLMs.

### Source Data

#### Data Sources & Licenses

The data samples used in the ClinicalBench benchmark are sourced from real clinical medical records of officially certified Grade 3A hospitals in China (Grade 3A hospitals are the highest level hospitals in China's "three-grade, six-class" classification system.). The collection of this data strictly adheres to the principles of patient privacy protection. No information related to the hospitals is disclosed. As detailed in Data Processing & Quality, to protect patient privacy, any personally identifiable information (PII) of patients, treatment regions, or other sensitive information has been manually identified and removed by the team of doctors. All data is obtained legally and ethically, and has been reviewed and approved by the Ethics Committees of the relevant hospitals, ensuring that research activities on these data comply with ethical and legal obligations. Supporting documents and certification materials from notary institutions, which demonstrate the legality and ethicality of our data collection process, can be found in the supplementary materials.

We are committed to responsible data management and strictly follow relevant laws and regulations involving the collection, use, and distribution of protected health information. To ensure the legal and regulated use of the dataset, we have formulated the **ClinicalBench Usage and Data Distribution License Agreement**, which can be found in the supplementary materials. This agreement strictly requires all users to use the data solely for research purposes and to adhere to strict regulations protecting patient privacy, prohibiting any form of personal information tracking or identification. Through these measures, we ensure the legality and ethics of data acquisition and use while supporting research that may promote the development of LLMs in clinical diagnostics.

#### Who are the source language producers?

Participating hospital doctors.

### Data Processing & Quality

#### Annotation process

The ClinicalBench benchmark is manually created by three senior clinicians and two AI researchers. The creation process covers 4 key steps, as follows.

- The **Data collection** step focuses on authenticity, diversity, privacy. Based on department divisions and common disease types in each department, the medical team selects representative real cases for each disease from the hospital case database with permission for research. Given that these clinical case data is the private information of hospitals, the risk of data leakage to any LLMs is completely eliminated.
- The **Professional knowledge review** step ensures the accuracy of the data. The team of doctors conducts a detailed professional review of the diagnostic information, treatment process, and results of each case to ensure the medical accuracy and proficiency of the data.
- The **Privacy protection and de-identification** step ensures privacy protection. To protect patient privacy, the team of doctors conducts two rounds of independent reviews to identify and remove any content that could reveal patient identities, treatment regions, or other sensitive information.
- The **Data integrity and compliance check** step aims for completeness and ethical compliance. Two AI researchers are responsible for reviewing the data to ensure that each record is complete, and meets the medical task requirements set in Section 3.4 of the paper. Additionally, they reconfirm that the dataset does not contain any sensitive information and strictly complies with the ethical guidelines.
  See the paper for more details..

#### Who are the annotators?

Three senior clinicians and two AI researchers. They are all authors of the paper.

### Personal and Sensitive Information

To protect patient privacy, any personally identifiable information (PII) of patients, treatment regions, or other sensitive information has been manually identified and removed by the team of doctors. For detailed processing methods, please refer to the explanation in the Annotation process section.

## Considerations for Using the Data

### Discussion of Biases

The data in ClinicalBench comes from mainland China and only follows the officially recommended diagnostic methods and procedures in mainland China. Therefore, there may be a lack of representativeness for other regions and countries.

## Additional Information

### Dataset Curators

The author list of the paper is an accurate list of specific contributors. Dataset is managed by Weixiang Yan.

### Licensing Information

- ClinicalBench is released under the ClinicalBench Usage and Data Distribution License Agreement. Please be sure to comply with the terms of use.
- The remaining code is released under the Apache-2.0 license. Please be sure to comply with the terms of use.

### Citation Information

Please cite the paper if you use the data or code from ClinicalLab.

```
@misc{yan2024clinicallab,
      title={ClinicalLab: Aligning Agents for Multi-Departmental Clinical Diagnostics in the Real World}, 
      author={Weixiang Yan and Haitian Liu and Tengxiao Wu and Qian Chen and Wen Wang and Haoyuan Chai and Jiayi Wang and Weishan Zhao and Yixin Zhang and Renjun Zhang and Li Zhu},
      year={2024},
      eprint={2406.13890},
      archivePrefix={arXiv}
}
```
