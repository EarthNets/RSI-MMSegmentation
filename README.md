# RSI-MMSegmentation

Official code for:
## GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data

The new version of the GAMUS dataset can be downloaded from the following [XShadow/GAMUS](https://huggingface.co/datasets/XShadow/GAMUS).

The old version of data containing DFC 2019: [link](https://syncandshare.lrz.de/dl/fiBpfqvv7QE3MxRC18Uocq/GAMUS.zip)

# Update
- [x] Remove the cities (OMA and JAX) from the DFC 2019 dataset to ensure the label quality
- [x] Add pytorch dataloader
- [ ] Simplify the codebase
- [ ] Re-run all experiments


 Comparison results of different multi-modal fusion methods on the RSMSS dataset for supervised semantic segmentation。

<div  align="center">    
 <img src="resources/cnn_fuse.png" width = "620" height = "620" alt="GAMUS" align=center />
</div>

<div  align="center">    
 <img src="resources/trans_fuse.png" width = "990" height = "400" alt="RSMSS" align=center />
</div>

  Comparison results of different multi-modal fusion strategies on the RSMSS dataset for supervised semantic segmentation.


# References
```
@article{xiong2023gamus,
  title={GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data},
  author={Xiong, Zhitong and Chen, Sining and Wang, Yi and Mou, Lichao and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2305.14914},
  year={2023}
}
```
