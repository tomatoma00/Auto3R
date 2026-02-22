# Auto3R
Code for Auto3R: Automated 3D Reconstruction and Scanning via Data-driven Uncertainty Quantification

## Pipline & Result
<div align="center">
  <img src="assets/3pipmainnew.png"/>
</div><br/>
    
---

## 0. Installation

You can simply follow the environment setup of [FisherRF](https://github.com/JiangWenPL/FisherRF).

Or install with the rnvironment.yml.

## 1. Pretrained weight

Please download from : https://westlakeu-my.sharepoint.com/:u:/g/personal/shenchentao_westlake_edu_cn/IQBd5rbRXlNDS48SyGYSNKr-AcWgVrwocQlldMjzpSvrb0M?e=lZqhkq

Then put it into the folder ssimruns

## 2. Usaged

Note: Due to the size limitation of the supplementary materials, we only provide pre-trained UQ for rendered images, and the UQ model for depth maps will be replaced by the HyperIQA (Su, Shaolin et.al. CVPR 2020), their pre-trained files is available.

## 3. Trained UQ in custom datasets

Use this command to run the demo:

```
bash scripts/demo.sh -s demodata -m output 
```

## 4. Acknowledgements
We are quite grateful for [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [FisherRF](https://github.com/JiangWenPL/FisherRF)

## cite
```
@misc{shen2025auto3rautomated3dreconstruction,
      title={Auto3R: Automated 3D Reconstruction and Scanning via Data-driven Uncertainty Quantification}, 
      author={Chentao Shen and Sizhe Zheng and Bingqian Wu and Yaohua Feng and Yuanchen Fei and Mingyu Mei and Hanwen Jiang and Xiangru Huang},
      year={2025},
      eprint={2512.04528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04528}, 
}
```
