# ClinicalLab: Aligning Agents for Multi-Departmental Clinical Diagnostics in the Real World

<div align="center">
    <a href=""><img src="./images/leaderboard.png">Website</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://arxiv.org/pdf/2406.13890">📄 Paper</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://forms.gle/Tkq5UTinW7bBB6388"><img src="./images/google_drive.png"> Apply for data access</a>
</div>

<br>

**ClinicalLab**, a comprehensive clinical diagnosis agent alignment suite. ClinicalLab includes **ClinicalBench**, an end-to-end multi-departmental clinical diagnostic evaluation benchmark for evaluating medical agents and LLMs. ClinicalBench is based on real cases that cover 24 departments and 150 diseases. ClinicalLab also includes four novel metrics (**ClinicalMetrics**) for evaluating the effectiveness of LLMs in clinical diagnostic tasks. **ClinicalAgent**, an end-to-end clinical agent that aligns with real-world clinical diagnostic practices. Our findings demonstrate the importance of aligning with modern medical practices in designing medical agents.

## 🌈 Update

* **[2024.06.19]** 🎉🎉🎉 ClinicalLab is published！🎉🎉🎉

## License

- ClinicalBench is released under the [ClinicalBench Usage and Data Distribution License Agreement](./DATA_LICENSE.pdf). Please be sure to comply with the terms of use.
- The remaining code is released under the [Apache-2.0 license](./CODE_LICENSE). Please be sure to comply with the terms of use.

## Datasets

In the appendix of our <a href="https://arxiv.org/pdf/2406.13890">paper</a> and the [data_examples](./data_examples) folder, we present examples from the ClinicalBench dataset, including both Chinese and English versions. Please note that **accessing the ClinicalBench dataset requires an application**. If you wish to access the full dataset, please read the [licensing documentation](./DATA_LICENSE.pdf) and submit an [access request](https://forms.gle/Tkq5UTinW7bBB6388). We will send the data to your specified email address within 48 hours.

For more information about the dataset, please refer to the [DATA CARD](./data_examples/DATASET_CARD.md).

## Dependencies

You can install everything all at once
```
conda create -n clinicallab python=3.9 -y
pip install -r requirements.txt
```

We also provide an [environment.yaml](./environment.yaml) file for your use.

## Code

Coming soon...




## Citation

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

## Contact

For questions, please feel free to reach out via email at ``yanweixiang.ywx@gmail.com``.


