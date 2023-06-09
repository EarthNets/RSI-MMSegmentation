# RSI-MMSegmentation

Official code for:
## GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data

The proposed benchmark dataset RSMSS can be donwloaded from the following [link](https://syncandshare.lrz.de/getlink/fi8rRALX7JwWtMaSH1jpxiVA/RSUSS.zip):
1. https://syncandshare.lrz.de/getlink/fi8rRALX7JwWtMaSH1jpxiVA/RSUSS.zip
2. https://mediatum.ub.tum.de/1661568
<div  align="center">    
 <img src="resources/RSMSS.png" width = "990" height = "270" alt="RSMSS" align=center />
</div>


# Add results of using only the height modality

<div  align="center">    
 <img src="resources/res1.png" width = "613" height = "460" alt="GAMUS" align=center />
</div>

  Comparison results of different multi-modal fusion methods on the RSMSS dataset for supervised semantic segmentation。

<div  align="center">    
 <img src="resources/res2.png" width = "620" height = "432" alt="GAMUS" align=center />
</div>

  Comparison results of different multi-modal fusion strategies on the RSMSS dataset for supervised semantic segmentation.

# Add visualization examples of different multi-modal fusion methods

<div  align="center">    
 <img src="resources/vis1.png" width = "800" height = "360" alt="RSMSS" align=center />
</div>

  updating more visualization results...

| Paradigm                             | Methods                             | Modality | Ground | Vegetation | Building | Water  | Road   | Tree   | mAcc   |
|--------------------------------------|-------------------------------------|----------|--------|------------|----------|--------|--------|--------|--------|
| \multirow{3}{*}{\makecell[c]{Single  |
|                                      | RTFNet                | RGB      | 0.7370 | 0.5980     | 0.8873   | 0.2144 | 0.6236 | 0.8666 | 0.6545 |
|                                      | FuseNet | RGB      | 0.3753 | 0.5104     | 0.8724   | 0.1045 | 0.6375 | 0.7887 | 0.5481 |
| \multirow{3}{*}{\makecell[c]{Single  |
|                                      | RTFNet                | Height   | 0.8223 | 0.5537     | 0.8708   | 0.2513 | 0.6764 | 0.8186 | 0.6655 |
|                                      | FuseNet  | Height   | 0.7821 | 0.4208     | 0.8173   | 0.6912 | 0.7455 | 0.4715 | 0.6547 |
| \multirow{4}{*}{\makecell[c]{Early   |
|                                      | RTFNet               | RGBH     | 0.7955 | 0.6521     | 0.8706   | 0.4695 | 0.6516 | 0.8135 | 0.7088 |
|                                      | FuseNet | RGBH     | 0.8084 | 0.4178     | 0.9049   | 0.6758 | 0.8179 | 0.6807 | 0.7176 |
| \multirow{4}{*}{\makecell[c]{Feature |
|                                      | RTFNet              | RGBH     | 0.7190 | 0.6732     | 0.8774   | 0.5975 | 0.7057 | 0.8533 | 0.7377 |
|                                      | FuseNet | RGBH     | 0.4716 | 0.7558     | 0.9109   | 0.3184 | 0.6221 | 0.9444 | 0.6705 |
| \multirow{3}{*}{\makecell[c]{Late    |
|                                      | RTFNet    | RGBH     | 0.7798 | 0.6195     | 0.8941   | 0.5141 | 0.6953 | 0.8087 | 0.7186 |
|                                      | FuseNet  | RGBH     | 0.7239 | 0.3185     | 0.9526   | 0.0758 | 0.8571 | 0.8194 | 0.6245 |

# Add experiments to study the transferability
<div  align="center">    
 <img src="resources/res3.png" width = "750" height = "150" alt="RSMSS" align=center />
</div>
We transfer the trained network weights from the proposed RSUSS dataset to the ISPRS Potsdam ataset. 
For the zero-shot transfer learning results, weights learned from our dataset can be directly transferred to the Potsdam dataset.
The results is no-doubt higher than that of weights from ImageNet. 
By fine-tuning the model on Potsdam dataset for 5 epochs, our results is clearly higher than weights from ImageNet. 

