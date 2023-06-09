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
 <img src="resources/res1.png" width = "613" height = "460" alt="RSMSS" align=center />
</div>

  Comparison results of different multi-modal fusion methods on the RSMSS dataset for supervised semantic segmentation。

<div  align="center">    
 <img src="resources/res2.png" width = "620" height = "432" alt="RSMSS" align=center />
</div>

  Comparison results of different multi-modal fusion strategies on the RSMSS dataset for supervised semantic segmentation.

# Add visualization examples of different multi-modal fusion methods

<div  align="center">    
 <img src="resources/vis1.png" width = "800" height = "360" alt="RSMSS" align=center />
</div>

  updating more visualization results...

# Add comparison and analysis of multiple modalities
<center  class="half">    
<a href="https://public.flourish.studio/visualisation/10968749/">
 <img src="resources/bar.png" width = "362" height = "250" alt="RSMSS" align=center />
 </a><a href="https://public.flourish.studio/visualisation/11006073/">
 <img src="resources/anay.png" width = "362" height = "250" alt="RSMSS" align=center />
 </a>
</center>

  Clicking on the pictures to view the data in an interactive way!

# Add experiments to study the transferability
<div  align="center">    
 <img src="resources/res3.png" width = "750" height = "150" alt="RSMSS" align=center />
</div>
We transfer the trained network weights from the proposed RSUSS dataset to the ISPRS Potsdam ataset. 
For the zero-shot transfer learning results, weights learned from our dataset can be directly transferred to the Potsdam dataset.
The results is no-doubt higher than that of weights from ImageNet. 
By fine-tuning the model on Potsdam dataset for 5 epochs, our results is clearly higher than weights from ImageNet. 

