# Model Zoo and Baselines

We provide a set of benchmark results and pre-trained models available for download. On PASCAL VOC and COCO, we provide results for seed 0, which are consistent with results from previous works. We also provide results over multiple seeds.

#### Available Models

We provide several models that are discussed in our paper.

- `Base Model`: model after first stage of base training.
- `FRCN+ft-full`: fine-tuning the entire model until convergence.
- `TFA w/ fc`: our approach with a FC-based classifier.
- `TFA w/ cos`: our approach with a cosine similarity based classifier.

#### Loading Models in Code

You can access the models from code using the fsdet.model_zoo API (official documentation [here](https://detectron2.readthedocs.io/modules/model_zoo.html)).

1. Pick a model and its config file from
  [model zoo](fsdet/model_zoo/model_zoo.py),
  for example, `COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml`.
2. ```python
   from fsdet import model_zoo
   model = model_zoo.get(
      "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml", trained=True)
   ```

## PASCAL VOC Object Detection Baselines

### Results on Seed 0

#### Base Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml">Base Model</a></td>
<td align="center">1</td>
<td align="center">54.9</td>
<td align="center">80.8</td>
<td align="center">61.1</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/base_model/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/base_model/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_base2.yaml">Base Model</a></td>
<td align="center">2</td>
<td align="center">55.1</td>
<td align="center">81.9</td>
<td align="center">61.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/base_model/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/base_model/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_base3.yaml">Base Model</a></td>
<td align="center">3</td>
<td align="center">55.6</td>
<td align="center">82.0</td>
<td align="center">61.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/base_model/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/base_model/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 1-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">34.0</td>
<td align="center">55.4</td>
<td align="center">36.7</td>
<td align="center">42.2</td>
<td align="center">68.9</td>
<td align="center">45.6</td>
<td align="center">9.3</td>
<td align="center">15.2</td>
<td align="center">10.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_1shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">43.8</td>
<td align="center">69.3</td>
<td align="center">47.6</td>
<td align="center">51.6</td>
<td align="center">80.2</td>
<td align="center">56.4</td>
<td align="center">20.3</td>
<td align="center">36.8</td>
<td align="center">21.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">44.0</td>
<td align="center">69.7</td>
<td align="center">47.8</td>
<td align="center">50.9</td>
<td align="center">79.6</td>
<td align="center">55.6</td>
<td align="center">23.4</td>
<td align="center">39.8</td>
<td align="center">24.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">30.1</td>
<td align="center">50.1</td>
<td align="center">31.9</td>
<td align="center">38.1</td>
<td align="center">62.4</td>
<td align="center">40.9</td>
<td align="center">6.0</td>
<td align="center">13.4</td>
<td align="center">4.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_1shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">41.2</td>
<td align="center">64.7</td>
<td align="center">45.3</td>
<td align="center">51.8</td>
<td align="center">80.3</td>
<td align="center">57.6</td>
<td align="center">9.6</td>
<td align="center">18.2</td>
<td align="center">8.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">41.1</td>
<td align="center">65.5</td>
<td align="center">44.7</td>
<td align="center">50.9</td>
<td align="center">79.5</td>
<td align="center">56.6</td>
<td align="center">11.7</td>
<td align="center">23.5</td>
<td align="center">9.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">36.5</td>
<td align="center">58.5</td>
<td align="center">39.9</td>
<td align="center">44.8</td>
<td align="center">71.4</td>
<td align="center">49.5</td>
<td align="center">11.6</td>
<td align="center">19.6</td>
<td align="center">11.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_1shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">43.0</td>
<td align="center">67.8</td>
<td align="center">45.9</td>
<td align="center">52.5</td>
<td align="center">81.1</td>
<td align="center">56.9</td>
<td align="center">14.3</td>
<td align="center">27.7</td>
<td align="center">12.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">42.3</td>
<td align="center">67.9</td>
<td align="center">44.8</td>
<td align="center">51.2</td>
<td align="center">80.3</td>
<td align="center">55.2</td>
<td align="center">15.6</td>
<td align="center">30.8</td>
<td align="center">13.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_1shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 2-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">34.8</td>
<td align="center">57.1</td>
<td align="center">36.3</td>
<td align="center">42.2</td>
<td align="center">69.4</td>
<td align="center">44.3</td>
<td align="center">12.5</td>
<td align="center">20.3</td>
<td align="center">12.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_2shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">41.1</td>
<td align="center">66.9</td>
<td align="center">43.2</td>
<td align="center">49.1</td>
<td align="center">79.5</td>
<td align="center">51.6</td>
<td align="center">17.2</td>
<td align="center">29.1</td>
<td align="center">17.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">42.2</td>
<td align="center">68.2</td>
<td align="center">45.5</td>
<td align="center">49.1</td>
<td align="center">78.9</td>
<td align="center">52.9</td>
<td align="center">21.5</td>
<td align="center">36.1</td>
<td align="center">23.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">32.5</td>
<td align="center">53.7</td>
<td align="center">34.2</td>
<td align="center">39.4</td>
<td align="center">64.8</td>
<td align="center">41.7</td>
<td align="center">11.7</td>
<td align="center">20.6</td>
<td align="center">11.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_2shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">40.1</td>
<td align="center">66.3</td>
<td align="center">42.0</td>
<td align="center">48.7</td>
<td align="center">78.7</td>
<td align="center">52.1</td>
<td align="center">14.3</td>
<td align="center">29.0</td>
<td align="center">11.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">39.4</td>
<td align="center">65.0</td>
<td align="center">42.1</td>
<td align="center">47.9</td>
<td align="center">77.7</td>
<td align="center">52.2</td>
<td align="center">14.0</td>
<td align="center">26.9</td>
<td align="center">12.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">35.4</td>
<td align="center">59.1</td>
<td align="center">36.8</td>
<td align="center">43.5</td>
<td align="center">71.8</td>
<td align="center">45.7</td>
<td align="center">11.1</td>
<td align="center">20.8</td>
<td align="center">10.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_2shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">42.7</td>
<td align="center">68.9</td>
<td align="center">45.0</td>
<td align="center">50.9</td>
<td align="center">80.6</td>
<td align="center">54.4</td>
<td align="center">18.1</td>
<td align="center">33.6</td>
<td align="center">16.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">42.4</td>
<td align="center">68.6</td>
<td align="center">44.7</td>
<td align="center">50.1</td>
<td align="center">79.9</td>
<td align="center">53.2</td>
<td align="center">19.4</td>
<td align="center">34.8</td>
<td align="center">19.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_2shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 3-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">34.5</td>
<td align="center">56.8</td>
<td align="center">37.2</td>
<td align="center">40.2</td>
<td align="center">66.1</td>
<td align="center">43.2</td>
<td align="center">17.6</td>
<td align="center">29.0</td>
<td align="center">19.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">44.6</td>
<td align="center">70.3</td>
<td align="center">49.4</td>
<td align="center">50.7</td>
<td align="center">79.2</td>
<td align="center">56.3</td>
<td align="center">26.2</td>
<td align="center">43.6</td>
<td align="center">28.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">45.0</td>
<td align="center">70.5</td>
<td align="center">50.4</td>
<td align="center">50.8</td>
<td align="center">79.1</td>
<td align="center">56.9</td>
<td align="center">27.8</td>
<td align="center">44.7</td>
<td align="center">30.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">32.5</td>
<td align="center">53.6</td>
<td align="center">34.4</td>
<td align="center">38.1</td>
<td align="center">62.0</td>
<td align="center">41.1</td>
<td align="center">15.7</td>
<td align="center">28.6</td>
<td align="center">14.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_3shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">42.0</td>
<td align="center">67.7</td>
<td align="center">44.8</td>
<td align="center">50.3</td>
<td align="center">79.2</td>
<td align="center">54.8</td>
<td align="center">17.1</td>
<td align="center">33.4</td>
<td align="center">14.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">42.2</td>
<td align="center">67.7</td>
<td align="center">45.8</td>
<td align="center">50.3</td>
<td align="center">78.8</td>
<td align="center">55.6</td>
<td align="center">17.9</td>
<td align="center">34.1</td>
<td align="center">16.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">36.5</td>
<td align="center">58.7</td>
<td align="center">39.0</td>
<td align="center">43.2</td>
<td align="center">68.7</td>
<td align="center">46.4</td>
<td align="center">16.3</td>
<td align="center">28.7</td>
<td align="center">16.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_3shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">45.9</td>
<td align="center">70.8</td>
<td align="center">50.0</td>
<td align="center">53.0</td>
<td align="center">80.3</td>
<td align="center">58.5</td>
<td align="center">24.5</td>
<td align="center">42.5</td>
<td align="center">24.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">45.7</td>
<td align="center">71.0</td>
<td align="center">50.0</td>
<td align="center">52.6</td>
<td align="center">80.4</td>
<td align="center">58.0</td>
<td align="center">25.0</td>
<td align="center">42.8</td>
<td align="center">26.1</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_3shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 5-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">37.4</td>
<td align="center">60.1</td>
<td align="center">40.3</td>
<td align="center">41.3</td>
<td align="center">66.7</td>
<td align="center">44.4</td>
<td align="center">25.5</td>
<td align="center">40.1</td>
<td align="center">27.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">46.7</td>
<td align="center">73.4</td>
<td align="center">51.1</td>
<td align="center">51.4</td>
<td align="center">79.2</td>
<td align="center">56.7</td>
<td align="center">32.4</td>
<td align="center">55.7</td>
<td align="center">34.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">47.2</td>
<td align="center">73.4</td>
<td align="center">52.3</td>
<td align="center">51.5</td>
<td align="center">79.3</td>
<td align="center">57.2</td>
<td align="center">34.2</td>
<td align="center">55.7</td>
<td align="center">37.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">32.6</td>
<td align="center">55.9</td>
<td align="center">33.5</td>
<td align="center">37.6</td>
<td align="center">63.7</td>
<td align="center">39.1</td>
<td align="center">17.5</td>
<td align="center">32.4</td>
<td align="center">16.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_5shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">43.0</td>
<td align="center">68.3</td>
<td align="center">46.5</td>
<td align="center">50.7</td>
<td align="center">79.3</td>
<td align="center">55.6</td>
<td align="center">19.9</td>
<td align="center">35.5</td>
<td align="center">19.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">43.1</td>
<td align="center">68.0</td>
<td align="center">47.2</td>
<td align="center">50.9</td>
<td align="center">78.9</td>
<td align="center">56.3</td>
<td align="center">19.9</td>
<td align="center">35.1</td>
<td align="center">19.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">38.1</td>
<td align="center">61.8</td>
<td align="center">40.4</td>
<td align="center">42.9</td>
<td align="center">68.3</td>
<td align="center">46.2</td>
<td align="center">23.4</td>
<td align="center">42.2</td>
<td align="center">22.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_5shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">46.3</td>
<td align="center">72.3</td>
<td align="center">50.4</td>
<td align="center">52.7</td>
<td align="center">80.2</td>
<td align="center">58.3</td>
<td align="center">27.1</td>
<td align="center">48.7</td>
<td align="center">26.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">46.3</td>
<td align="center">72.5</td>
<td align="center">49.9</td>
<td align="center">52.5</td>
<td align="center">80.2</td>
<td align="center">56.9</td>
<td align="center">27.9</td>
<td align="center">49.5</td>
<td align="center">28.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_5shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 10-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">37.5</td>
<td align="center">60.9</td>
<td align="center">40.0</td>
<td align="center">40.4</td>
<td align="center">66.0</td>
<td align="center">42.9</td>
<td align="center">28.7</td>
<td align="center">45.5</td>
<td align="center">31.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/FRCN+ft-full_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">46.6</td>
<td align="center">73.2</td>
<td align="center">51.0</td>
<td align="center">50.9</td>
<td align="center">78.6</td>
<td align="center">56.1</td>
<td align="center">33.5</td>
<td align="center">57.0</td>
<td align="center">35.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_fc_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">47.3</td>
<td align="center">72.8</td>
<td align="center">52.2</td>
<td align="center">51.5</td>
<td align="center">78.4</td>
<td align="center">56.8</td>
<td align="center">34.5</td>
<td align="center">56.0</td>
<td align="center">38.1</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">32.2</td>
<td align="center">55.5</td>
<td align="center">33.3</td>
<td align="center">36.0</td>
<td align="center">61.0</td>
<td align="center">37.7</td>
<td align="center">20.8</td>
<td align="center">38.8</td>
<td align="center">20.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/FRCN+ft-full_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_10shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">42.7</td>
<td align="center">68.7</td>
<td align="center">45.9</td>
<td align="center">50.1</td>
<td align="center">78.6</td>
<td align="center">55.0</td>
<td align="center">20.5</td>
<td align="center">39.0</td>
<td align="center">18.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_fc_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">43.3</td>
<td align="center">68.6</td>
<td align="center">47.2</td>
<td align="center">50.7</td>
<td align="center">78.5</td>
<td align="center">56.1</td>
<td align="center">21.1</td>
<td align="center">39.1</td>
<td align="center">20.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split2/tfa_cos_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">37.3</td>
<td align="center">60.8</td>
<td align="center">39.4</td>
<td align="center">41.3</td>
<td align="center">67.0</td>
<td align="center">43.4</td>
<td align="center">25.2</td>
<td align="center">42.1</td>
<td align="center">27.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/FRCN+ft-full_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_10shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">46.7</td>
<td align="center">72.2</td>
<td align="center">51.2</td>
<td align="center">52.6</td>
<td align="center">79.5</td>
<td align="center">58.2</td>
<td align="center">29.1</td>
<td align="center">50.2</td>
<td align="center">30.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">47.0</td>
<td align="center">72.4</td>
<td align="center">51.7</td>
<td align="center">52.8</td>
<td align="center">79.9</td>
<td align="center">58.5</td>
<td align="center">29.4</td>
<td align="center">49.8</td>
<td align="center">31.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_10shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

### Results Over 30 Seeds

#### 1-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">30.2</td>
<td align="center">49.4</td>
<td align="center">32.2</td>
<td align="center">38.2</td>
<td align="center">62.6</td>
<td align="center">40.8</td>
<td align="center">6</td>
<td align="center">9.9</td>
<td align="center">6.3</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_1shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">39.6</td>
<td align="center">63.5</td>
<td align="center">43.2</td>
<td align="center">48.7</td>
<td align="center">77.1</td>
<td align="center">53.7</td>
<td align="center">12.2</td>
<td align="center">22.9</td>
<td align="center">11.6</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">40.6</td>
<td align="center">64.5</td>
<td align="center">44.7</td>
<td align="center">49.4</td>
<td align="center">77.6</td>
<td align="center">54.8</td>
<td align="center">14.2</td>
<td align="center">25.3</td>
<td align="center">14.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">30.3</td>
<td align="center">49.7</td>
<td align="center">32.3</td>
<td align="center">38.8</td>
<td align="center">63.2</td>
<td align="center">41.6</td>
<td align="center">5</td>
<td align="center">9.4</td>
<td align="center">4.5</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_1shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">36.2</td>
<td align="center">59.6</td>
<td align="center">38.7</td>
<td align="center">45.6</td>
<td align="center">73.8</td>
<td align="center">49.4</td>
<td align="center">8.1</td>
<td align="center">16.9</td>
<td align="center">6.6</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">36.7</td>
<td align="center">59.9</td>
<td align="center">39.3</td>
<td align="center">45.9</td>
<td align="center">73.8</td>
<td align="center">49.8</td>
<td align="center">9.0</td>
<td align="center">18.3</td>
<td align="center">7.8</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">30.8</td>
<td align="center">49.8</td>
<td align="center">32.9</td>
<td align="center">39.6</td>
<td align="center">63.7</td>
<td align="center">42.5</td>
<td align="center">4.5</td>
<td align="center">8.1</td>
<td align="center">4.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_1shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">39.0</td>
<td align="center">62.3</td>
<td align="center">42.1</td>
<td align="center">49.5</td>
<td align="center">77.8</td>
<td align="center">54.0</td>
<td align="center">7.8</td>
<td align="center">15.7</td>
<td align="center">6.5</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">40.1</td>
<td align="center">63.5</td>
<td align="center">43.6</td>
<td align="center">50.2</td>
<td align="center">78.7</td>
<td align="center">55.1</td>
<td align="center">9.6</td>
<td align="center">17.9</td>
<td align="center">9.1</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 2-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">30.5</td>
<td align="center">49.4</td>
<td align="center">32.6</td>
<td align="center">37.3</td>
<td align="center">60.7</td>
<td align="center">40.1</td>
<td align="center">9.9</td>
<td align="center">15.6</td>
<td align="center">10.3</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_2shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">40.5</td>
<td align="center">65.5</td>
<td align="center">43.8</td>
<td align="center">47.8</td>
<td align="center">75.8</td>
<td align="center">52.2</td>
<td align="center">18.9</td>
<td align="center">34.5</td>
<td align="center">18.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">42.6</td>
<td align="center">67.1</td>
<td align="center">47.0</td>
<td align="center">49.6</td>
<td align="center">77.3</td>
<td align="center">55.0</td>
<td align="center">21.7</td>
<td align="center">36.4</td>
<td align="center">22.8</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">30.7</td>
<td align="center">49.7</td>
<td align="center">32.9</td>
<td align="center">38.4</td>
<td align="center">61.6</td>
<td align="center">41.4</td>
<td align="center">7.7</td>
<td align="center">13.8</td>
<td align="center">7.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_2shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">38.5</td>
<td align="center">62.8</td>
<td align="center">41.2</td>
<td align="center">46.9</td>
<td align="center">74.9</td>
<td align="center">51.2</td>
<td align="center">13.1</td>
<td align="center">26.4</td>
<td align="center">11.3</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">39.0</td>
<td align="center">63.0</td>
<td align="center">42.1</td>
<td align="center">47.3</td>
<td align="center">74.9</td>
<td align="center">51.9</td>
<td align="center">14.1</td>
<td align="center">27.5</td>
<td align="center">12.7</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">31.3</td>
<td align="center">50.2</td>
<td align="center">33.5</td>
<td align="center">39.1</td>
<td align="center">62.4</td>
<td align="center">42</td>
<td align="center">8</td>
<td align="center">13.9</td>
<td align="center">7.9</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_2shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">41.1</td>
<td align="center">65.1</td>
<td align="center">44.3</td>
<td align="center">50.1</td>
<td align="center">77.7</td>
<td align="center">54.8</td>
<td align="center">14.2</td>
<td align="center">27.2</td>
<td align="center">12.6</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">41.8</td>
<td align="center">65.6</td>
<td align="center">45.3</td>
<td align="center">50.7</td>
<td align="center">78.4</td>
<td align="center">55.6</td>
<td align="center">15.1</td>
<td align="center">27.2</td>
<td align="center">14.4</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 3-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">31.8</td>
<td align="center">51.4</td>
<td align="center">34.2</td>
<td align="center">37.9</td>
<td align="center">61.3</td>
<td align="center">40.7</td>
<td align="center">13.7</td>
<td align="center">21.6</td>
<td align="center">14.8</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">41.8</td>
<td align="center">67.1</td>
<td align="center">45.4</td>
<td align="center">48.2</td>
<td align="center">76.0</td>
<td align="center">53.1</td>
<td align="center">22.6</td>
<td align="center">40.4</td>
<td align="center">22.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">43.7</td>
<td align="center">68.5</td>
<td align="center">48.3</td>
<td align="center">49.8</td>
<td align="center">77.3</td>
<td align="center">55.4</td>
<td align="center">25.4</td>
<td align="center">42.1</td>
<td align="center">27.0</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">31.1</td>
<td align="center">50.1</td>
<td align="center">33.2</td>
<td align="center">38.1</td>
<td align="center">61</td>
<td align="center">41.2</td>
<td align="center">9.8</td>
<td align="center">17.4</td>
<td align="center">9.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_3shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">39.4</td>
<td align="center">64.2</td>
<td align="center">42.0</td>
<td align="center">47.5</td>
<td align="center">75.4</td>
<td align="center">51.7</td>
<td align="center">15.2</td>
<td align="center">30.5</td>
<td align="center">13.1</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">40.1</td>
<td align="center">64.5</td>
<td align="center">43.3</td>
<td align="center">48.1</td>
<td align="center">75.6</td>
<td align="center">52.9</td>
<td align="center">16.0</td>
<td align="center">30.9</td>
<td align="center">14.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">32.1</td>
<td align="center">51.3</td>
<td align="center">34.3</td>
<td align="center">39.1</td>
<td align="center">62.1</td>
<td align="center">42.1</td>
<td align="center">11.1</td>
<td align="center">19</td>
<td align="center">11.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_3shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">40.4</td>
<td align="center">65.4</td>
<td align="center">43.1</td>
<td align="center">47.8</td>
<td align="center">75.6</td>
<td align="center">52.1</td>
<td align="center">18.1</td>
<td align="center">34.7</td>
<td align="center">16.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">43.1</td>
<td align="center">67.5</td>
<td align="center">46.7</td>
<td align="center">51.1</td>
<td align="center">78.6</td>
<td align="center">56.3</td>
<td align="center">18.9</td>
<td align="center">34.3</td>
<td align="center">18.1</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 5-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">32.7</td>
<td align="center">52.5</td>
<td align="center">35</td>
<td align="center">37.6</td>
<td align="center">60.6</td>
<td align="center">40.3</td>
<td align="center">17.9</td>
<td align="center">28</td>
<td align="center">19.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">41.9</td>
<td align="center">68.0</td>
<td align="center">45.0</td>
<td align="center">47.2</td>
<td align="center">75.1</td>
<td align="center">51.5</td>
<td align="center">25.9</td>
<td align="center">46.7</td>
<td align="center">25.3</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">44.8</td>
<td align="center">70.1</td>
<td align="center">49.4</td>
<td align="center">50.1</td>
<td align="center">77.4</td>
<td align="center">55.6</td>
<td align="center">28.9</td>
<td align="center">47.9</td>
<td align="center">30.6</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">31.5</td>
<td align="center">50.8</td>
<td align="center">33.6</td>
<td align="center">37.9</td>
<td align="center">60.4</td>
<td align="center">40.8</td>
<td align="center">12.4</td>
<td align="center">21.9</td>
<td align="center">12.1</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_5shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">40.0</td>
<td align="center">65.1</td>
<td align="center">42.6</td>
<td align="center">47.5</td>
<td align="center">75.3</td>
<td align="center">51.6</td>
<td align="center">17.5</td>
<td align="center">34.6</td>
<td align="center">15.5</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">40.9</td>
<td align="center">65.7</td>
<td align="center">44.1</td>
<td align="center">48.6</td>
<td align="center">76.2</td>
<td align="center">53.3</td>
<td align="center">17.8</td>
<td align="center">34.1</td>
<td align="center">16.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">32.4</td>
<td align="center">51.7</td>
<td align="center">34.4</td>
<td align="center">38.5</td>
<td align="center">61</td>
<td align="center">41.3</td>
<td align="center">14</td>
<td align="center">23.9</td>
<td align="center">13.7</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_5shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">41.3</td>
<td align="center">67.1</td>
<td align="center">44.0</td>
<td align="center">48.0</td>
<td align="center">75.8</td>
<td align="center">52.2</td>
<td align="center">21.4</td>
<td align="center">40.8</td>
<td align="center">19.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">44.1</td>
<td align="center">69.1</td>
<td align="center">47.8</td>
<td align="center">51.3</td>
<td align="center">78.5</td>
<td align="center">56.4</td>
<td align="center">22.8</td>
<td align="center">40.8</td>
<td align="center">22.1</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 10-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">1</td>
<td align="center">33.3</td>
<td align="center">53.8</td>
<td align="center">35.5</td>
<td align="center">36.8</td>
<td align="center">59.8</td>
<td align="center">39.2</td>
<td align="center">22.7</td>
<td align="center">35.6</td>
<td align="center">24.4</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml">TFA w/ fc</a></td>
<td align="center">1</td>
<td align="center">42.8</td>
<td align="center">69.5</td>
<td align="center">46.0</td>
<td align="center">47.3</td>
<td align="center">75.4</td>
<td align="center">51.6</td>
<td align="center">29.3</td>
<td align="center">52.0</td>
<td align="center">29.0</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml">TFA w/ cos</a></td>
<td align="center">1</td>
<td align="center">45.8</td>
<td align="center">71.3</td>
<td align="center">50.4</td>
<td align="center">50.4</td>
<td align="center">77.5</td>
<td align="center">55.9</td>
<td align="center">32.0</td>
<td align="center">52.9</td>
<td align="center">33.7</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">2</td>
<td align="center">32.2</td>
<td align="center">52.3</td>
<td align="center">34.1</td>
<td align="center">37.2</td>
<td align="center">59.8</td>
<td align="center">39.9</td>
<td align="center">17</td>
<td align="center">29.8</td>
<td align="center">16.7</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_10shot.yaml">TFA w/ fc</a></td>
<td align="center">2</td>
<td align="center">41.3</td>
<td align="center">67.0</td>
<td align="center">44.0</td>
<td align="center">48.3</td>
<td align="center">76.1</td>
<td align="center">52.7</td>
<td align="center">20.2</td>
<td align="center">39.7</td>
<td align="center">18.0</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml">TFA w/ cos</a></td>
<td align="center">2</td>
<td align="center">42.3</td>
<td align="center">67.6</td>
<td align="center">45.7</td>
<td align="center">49.4</td>
<td align="center">76.9</td>
<td align="center">54.5</td>
<td align="center">20.8</td>
<td align="center">39.5</td>
<td align="center">19.2</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">3</td>
<td align="center">33.1</td>
<td align="center">53.1</td>
<td align="center">35.2</td>
<td align="center">38</td>
<td align="center">60.5</td>
<td align="center">40.7</td>
<td align="center">18.4</td>
<td align="center">31</td>
<td align="center">18.7</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_10shot.yaml">TFA w/ fc</a></td>
<td align="center">3</td>
<td align="center">42.2</td>
<td align="center">68.3</td>
<td align="center">44.9</td>
<td align="center">48.5</td>
<td align="center">76.2</td>
<td align="center">52.9</td>
<td align="center">23.3</td>
<td align="center">44.6</td>
<td align="center">21.0</td>
</tr>

<tr><td align="left"><a href="configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml">TFA w/ cos</a></td>
<td align="center">3</td>
<td align="center">45.0</td>
<td align="center">70.3</td>
<td align="center">48.9</td>
<td align="center">51.6</td>
<td align="center">78.6</td>
<td align="center">57.0</td>
<td align="center">25.4</td>
<td align="center">45.6</td>
<td align="center">24.7</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

## COCO Object Detection Baselines

### Results on Seed 0

#### Base Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml">Base Model</a></td>
<td align="center">39.2</td>
<td align="center">59.3</td>
<td align="center">42.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/base_model/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/base_model/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 1-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">19.1</td>
<td align="center">30.6</td>
<td align="center">21.1</td>
<td align="center">24.8</td>
<td align="center">39.7</td>
<td align="center">27.4</td>
<td align="center">1.8</td>
<td align="center">3.2</td>
<td align="center">2.1</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml">TFA w/ fc</a></td>
<td align="center">25.7</td>
<td align="center">41.0</td>
<td align="center">27.7</td>
<td align="center">33.3</td>
<td align="center">52.8</td>
<td align="center">36.0</td>
<td align="center">2.9</td>
<td align="center">5.7</td>
<td align="center">2.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_1shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml">TFA w/ cos</a></td>
<td align="center">26.4</td>
<td align="center">42.5</td>
<td align="center">28.2</td>
<td align="center">34.1</td>
<td align="center">54.7</td>
<td align="center">36.4</td>
<td align="center">3.4</td>
<td align="center">5.8</td>
<td align="center">3.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_1shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_1shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 2-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">18.9</td>
<td align="center">29.9</td>
<td align="center">20.7</td>
<td align="center">24.2</td>
<td align="center">37.9</td>
<td align="center">26.6</td>
<td align="center">3.1</td>
<td align="center">5.8</td>
<td align="center">2.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml">TFA w/ fc</a></td>
<td align="center">26.1</td>
<td align="center">41.4</td>
<td align="center">28.3</td>
<td align="center">33.3</td>
<td align="center">52.3</td>
<td align="center">36.4</td>
<td align="center">4.3</td>
<td align="center">8.5</td>
<td align="center">4.1</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_2shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot.yaml">TFA w/ cos</a></td>
<td align="center">27.1</td>
<td align="center">43.4</td>
<td align="center">29.4</td>
<td align="center">34.7</td>
<td align="center">55.1</td>
<td align="center">37.6</td>
<td align="center">4.6</td>
<td align="center">8.3</td>
<td align="center">4.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_2shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_2shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 3-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">18.3</td>
<td align="center">29.5</td>
<td align="center">19.8</td>
<td align="center">22.9</td>
<td align="center">36.3</td>
<td align="center">25.0</td>
<td align="center">4.7</td>
<td align="center">8.9</td>
<td align="center">4.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml">TFA w/ fc</a></td>
<td align="center">26.9</td>
<td align="center">42.5</td>
<td align="center">29.5</td>
<td align="center">33.7</td>
<td align="center">52.5</td>
<td align="center">37.2</td>
<td align="center">6.7</td>
<td align="center">12.6</td>
<td align="center">6.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_3shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot.yaml">TFA w/ cos</a></td>
<td align="center">27.7</td>
<td align="center">44.1</td>
<td align="center">30.0</td>
<td align="center">34.7</td>
<td align="center">54.8</td>
<td align="center">37.9</td>
<td align="center">6.6</td>
<td align="center">12.1</td>
<td align="center">6.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_3shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_3shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 5-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">18.0</td>
<td align="center">29.0</td>
<td align="center">19.3</td>
<td align="center">22.0</td>
<td align="center">34.9</td>
<td align="center">23.8</td>
<td align="center">6.0</td>
<td align="center">11.3</td>
<td align="center">5.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml">TFA w/ fc</a></td>
<td align="center">27.5</td>
<td align="center">43.6</td>
<td align="center">30.0</td>
<td align="center">33.9</td>
<td align="center">52.8</td>
<td align="center">37.2</td>
<td align="center">8.4</td>
<td align="center">16.0</td>
<td align="center">8.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_5shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml">TFA w/ cos</a></td>
<td align="center">28.1</td>
<td align="center">44.6</td>
<td align="center">30.2</td>
<td align="center">34.7</td>
<td align="center">54.4</td>
<td align="center">37.6</td>
<td align="center">8.3</td>
<td align="center">15.3</td>
<td align="center">8.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_5shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_5shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 10-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">18.1</td>
<td align="center">30.0</td>
<td align="center">18.9</td>
<td align="center">21.0</td>
<td align="center">34.3</td>
<td align="center">22.1</td>
<td align="center">9.2</td>
<td align="center">17.0</td>
<td align="center">9.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml">TFA w/ fc</a></td>
<td align="center">27.9</td>
<td align="center">44.6</td>
<td align="center">30.4</td>
<td align="center">33.9</td>
<td align="center">53.1</td>
<td align="center">37.4</td>
<td align="center">10.0</td>
<td align="center">19.2</td>
<td align="center">9.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_10shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml">TFA w/ cos</a></td>
<td align="center">28.7</td>
<td align="center">46.0</td>
<td align="center">31.0</td>
<td align="center">35.0</td>
<td align="center">55.0</td>
<td align="center">38.3</td>
<td align="center">10.0</td>
<td align="center">19.1</td>
<td align="center">9.3</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_10shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_10shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 30-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">18.6</td>
<td align="center">30.8</td>
<td align="center">19.3</td>
<td align="center">20.6</td>
<td align="center">33.4</td>
<td align="center">21.7</td>
<td align="center">12.5</td>
<td align="center">23.0</td>
<td align="center">12.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_30shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/FRCN+ft-full_30shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_30shot.yaml">TFA w/ fc</a></td>
<td align="center">29.7</td>
<td align="center">46.8</td>
<td align="center">32.5</td>
<td align="center">35.1</td>
<td align="center">54.2</td>
<td align="center">38.9</td>
<td align="center">13.4</td>
<td align="center">24.7</td>
<td align="center">13.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_30shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_fc_30shot/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml">TFA w/ cos</a></td>
<td align="center">30.3</td>
<td align="center">47.9</td>
<td align="center">32.9</td>
<td align="center">35.8</td>
<td align="center">55.5</td>
<td align="center">39.4</td>
<td align="center">13.7</td>
<td align="center">24.9</td>
<td align="center">13.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_30shot/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/coco/tfa_cos_30shot/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

### Results Over 10 Seeds

#### 1-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">16.2</td>
<td align="center">25.8</td>
<td align="center">17.6</td>
<td align="center">21</td>
<td align="center">33.3</td>
<td align="center">23</td>
<td align="center">1.7</td>
<td align="center">3.3</td>
<td align="center">1.6</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml">TFA w/ fc</a></td>
<td align="center">24</td>
<td align="center">38.9</td>
<td align="center">25.8</td>
<td align="center">31.5</td>
<td align="center">50.7</td>
<td align="center">33.9</td>
<td align="center">1.6</td>
<td align="center">3.4</td>
<td align="center">1.3</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml">TFA w/ cos</a></td>
<td align="center">24.4</td>
<td align="center">39.8</td>
<td align="center">26.1</td>
<td align="center">31.9</td>
<td align="center">51.8</td>
<td align="center">34.3</td>
<td align="center">1.9</td>
<td align="center">3.8</td>
<td align="center">1.7</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 2-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">15.8</td>
<td align="center">25</td>
<td align="center">17.3</td>
<td align="center">20</td>
<td align="center">31.4</td>
<td align="center">22.2</td>
<td align="center">3.1</td>
<td align="center">6.1</td>
<td align="center">2.9</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml">TFA w/ fc</a></td>
<td align="center">24.5</td>
<td align="center">39.3</td>
<td align="center">26.5</td>
<td align="center">31.4</td>
<td align="center">49.8</td>
<td align="center">34.3</td>
<td align="center">3.8</td>
<td align="center">7.8</td>
<td align="center">3.2</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot.yaml">TFA w/ cos</a></td>
<td align="center">24.9</td>
<td align="center">40.1</td>
<td align="center">27</td>
<td align="center">31.9</td>
<td align="center">50.8</td>
<td align="center">34.8</td>
<td align="center">3.9</td>
<td align="center">7.8</td>
<td align="center">3.6</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 3-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">15</td>
<td align="center">23.9</td>
<td align="center">16.4</td>
<td align="center">18.8</td>
<td align="center">29.5</td>
<td align="center">20.7</td>
<td align="center">3.7</td>
<td align="center">7.1</td>
<td align="center">3.5</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml">TFA w/ fc</a></td>
<td align="center">24.9</td>
<td align="center">39.7</td>
<td align="center">27.1</td>
<td align="center">31.5</td>
<td align="center">49.6</td>
<td align="center">34.6</td>
<td align="center">5</td>
<td align="center">9.9</td>
<td align="center">4.6</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot.yaml">TFA w/ cos</a></td>
<td align="center">25.3</td>
<td align="center">40.4</td>
<td align="center">27.6</td>
<td align="center">32</td>
<td align="center">50.5</td>
<td align="center">35.1</td>
<td align="center">5.1</td>
<td align="center">9.9</td>
<td align="center">4.8</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 5-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">14.4</td>
<td align="center">23</td>
<td align="center">15.6</td>
<td align="center">17.6</td>
<td align="center">27.8</td>
<td align="center">19.3</td>
<td align="center">4.6</td>
<td align="center">8.7</td>
<td align="center">4.4</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml">TFA w/ fc</a></td>
<td align="center">25.6</td>
<td align="center">40.7</td>
<td align="center">28</td>
<td align="center">31.8</td>
<td align="center">49.8</td>
<td align="center">35.2</td>
<td align="center">6.9</td>
<td align="center">13.4</td>
<td align="center">6.3</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml">TFA w/ cos</a></td>
<td align="center">25.9</td>
<td align="center">41.2</td>
<td align="center">28.4</td>
<td align="center">32.3</td>
<td align="center">50.5</td>
<td align="center">35.6</td>
<td align="center">7</td>
<td align="center">13.3</td>
<td align="center">6.5</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 10-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">13.4</td>
<td align="center">21.8</td>
<td align="center">14.5</td>
<td align="center">16.1</td>
<td align="center">25.7</td>
<td align="center">17.5</td>
<td align="center">5.5</td>
<td align="center">10</td>
<td align="center">5.5</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml">TFA w/ fc</a></td>
<td align="center">26.2</td>
<td align="center">41.8</td>
<td align="center">28.6</td>
<td align="center">32</td>
<td align="center">49.9</td>
<td align="center">35.3</td>
<td align="center">9.1</td>
<td align="center">17.3</td>
<td align="center">8.5</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml">TFA w/ cos</a></td>
<td align="center">26.6</td>
<td align="center">42.2</td>
<td align="center">29</td>
<td align="center">32.4</td>
<td align="center">50.6</td>
<td align="center">35.7</td>
<td align="center">9.1</td>
<td align="center">17.1</td>
<td align="center">8.8</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### 30-shot

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>bAP</th>
<th valign="bottom"><br/>bAP50</th>
<th valign="bottom"><br/>bAP75</th>
<th valign="bottom"><br/>nAP</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom"><br/>nAP75</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot_unfreeze.yaml">FRCN+ft-full</a></td>
<td align="center">13.5</td>
<td align="center">21.8</td>
<td align="center">14.5</td>
<td align="center">15.6</td>
<td align="center">24.8</td>
<td align="center">16.9</td>
<td align="center">7.4</td>
<td align="center">13.1</td>
<td align="center">7.4</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_30shot.yaml">TFA w/ fc</a></td>
<td align="center">28.4</td>
<td align="center">44.4</td>
<td align="center">31.2</td>
<td align="center">33.8</td>
<td align="center">51.8</td>
<td align="center">37.6</td>
<td align="center">12</td>
<td align="center">22.2</td>
<td align="center">11.8</td>
</tr>

<tr><td align="left"><a href="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml">TFA w/ cos</a></td>
<td align="center">28.7</td>
<td align="center">44.7</td>
<td align="center">31.5</td>
<td align="center">34.2</td>
<td align="center">52.3</td>
<td align="center">38</td>
<td align="center">12.1</td>
<td align="center">22</td>
<td align="center">12</td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

## LVIS Object Detection Baselines

### Base Models

#### With Repeat Sampling

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>APs</th>
<th valign="bottom"><br/>APm</th>
<th valign="bottom"><br/>APl</th>
<th valign="bottom"><br/>APc</th>
<th valign="bottom"><br/>APf</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_base.yaml">Base Model w/ fc</a></td>
<td align="center">R-50</td>
<td align="center">20.5</td>
<td align="center">34.5</td>
<td align="center">21.5</td>
<td align="center">17.3</td>
<td align="center">26.0</td>
<td align="center">33.1</td>
<td align="center">21.3</td>
<td align="center">27.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_repeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_repeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_base_cosine.yaml">Base Model w/ cos</a></td>
<td align="center">R-50</td>
<td align="center">20.2</td>
<td align="center">33.5</td>
<td align="center">20.9</td>
<td align="center">17.4</td>
<td align="center">25.7</td>
<td align="center">32.8</td>
<td align="center">20.4</td>
<td align="center">27.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_repeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_repeat_cos/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_base.yaml">Base Model w/ fc</a></td>
<td align="center">R-101</td>
<td align="center">22.3</td>
<td align="center">36.8</td>
<td align="center">23.6</td>
<td align="center">17.9</td>
<td align="center">28.6</td>
<td align="center">35.5</td>
<td align="center">24.0</td>
<td align="center">29.2</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_repeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_repeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_base_cosine.yaml">Base Model w/ cos</a></td>
<td align="center">R-101</td>
<td align="center">21.7</td>
<td align="center">35.6</td>
<td align="center">22.7</td>
<td align="center">17.7</td>
<td align="center">18.4</td>
<td align="center">34.9</td>
<td align="center">22.3</td>
<td align="center">29.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_repeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_repeat_cos/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### No Repeat Sampling

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>APs</th>
<th valign="bottom"><br/>APm</th>
<th valign="bottom"><br/>APl</th>
<th valign="bottom"><br/>APc</th>
<th valign="bottom"><br/>APf</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_base_norepeat.yaml">Base Model w/ fc</a></td>
<td align="center">R-50</td>
<td align="center">18.6</td>
<td align="center">31.6</td>
<td align="center">19.2</td>
<td align="center">16.6</td>
<td align="center">24.2</td>
<td align="center">31.8</td>
<td align="center">16.6</td>
<td align="center">28.5</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_norepeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_norepeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_base_norepeat_cosine.yaml">Base Model w/ cos</a></td>
<td align="center">R-50</td>
<td align="center">18.0</td>
<td align="center">30.1</td>
<td align="center">18.7</td>
<td align="center">16.0</td>
<td align="center">23.9</td>
<td align="center">31.4</td>
<td align="center">15.3</td>
<td align="center">28.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_norepeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_base_norepeat_cos/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_base_norepeat.yaml">Base Model w/ fc</a></td>
<td align="center">R-101</td>
<td align="center">19.8</td>
<td align="center">32.7</td>
<td align="center">21.0</td>
<td align="center">17.4</td>
<td align="center">26.3</td>
<td align="center">34.0</td>
<td align="center">18.2</td>
<td align="center">29.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_norepeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_norepeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_base_norepeat_cosine.yaml">Base Model w/ cos</a></td>
<td align="center">R-101</td>
<td align="center">19.8</td>
<td align="center">32.6</td>
<td align="center">20.6</td>
<td align="center">17.6</td>
<td align="center">25.6</td>
<td align="center">34.2</td>
<td align="center">17.9</td>
<td align="center">30.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_norepeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_base_norepeat_cos/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

### Fine-tuned Models

#### With Repeat Sampling

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>APs</th>
<th valign="bottom"><br/>APm</th>
<th valign="bottom"><br/>APl</th>
<th valign="bottom"><br/>APr</th>
<th valign="bottom"><br/>APc</th>
<th valign="bottom"><br/>APf</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_combined_all.yaml">TFA w/ fc</a></td>
<td align="center">R-50</td>
<td align="center">24.1</td>
<td align="center">39.9</td>
<td align="center">25.4</td>
<td align="center">19.5</td>
<td align="center">29.1</td>
<td align="center">36.7</td>
<td align="center">14.9</td>
<td align="center">23.9</td>
<td align="center">27.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_repeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_repeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_cosine_combined_all.yaml">TFA w/ cos</a></td>
<td align="center">R-50</td>
<td align="center">24.4</td>
<td align="center">40.0</td>
<td align="center">26.1</td>
<td align="center">19.9</td>
<td align="center">29.5</td>
<td align="center">38.2</td>
<td align="center">16.9</td>
<td align="center">24.3</td>
<td align="center">27.7</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_repeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_repeat_cos/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_combined_all.yaml">TFA w/ fc</a></td>
<td align="center">R-101</td>
<td align="center">25.4</td>
<td align="center">41.8</td>
<td align="center">27.0</td>
<td align="center">19.8</td>
<td align="center">31.1</td>
<td align="center">39.2</td>
<td align="center">15.5</td>
<td align="center">26.0</td>
<td align="center">28.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_repeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_repeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all.yaml">TFA w/ cos</a></td>
<td align="center">R-101</td>
<td align="center">26.2</td>
<td align="center">41.8</td>
<td align="center">27.5</td>
<td align="center">20.2</td>
<td align="center">32.0</td>
<td align="center">39.9</td>
<td align="center">17.3</td>
<td align="center">26.4</td>
<td align="center">29.6</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_repeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_repeat_cos/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

#### No Repeat Sampling

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"><br/>AP</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom"><br/>AP75</th>
<th valign="bottom"><br/>APs</th>
<th valign="bottom"><br/>APm</th>
<th valign="bottom"><br/>APl</th>
<th valign="bottom"><br/>APr</th>
<th valign="bottom"><br/>APc</th>
<th valign="bottom"><br/>APf</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_combined_all_norepeat.yaml">TFA w/ fc</a></td>
<td align="center">R-50</td>
<td align="center">22.3</td>
<td align="center">37.8</td>
<td align="center">22.2</td>
<td align="center">18.5</td>
<td align="center">28.2</td>
<td align="center">36.6</td>
<td align="center">14.3</td>
<td align="center">21.1</td>
<td align="center">27.0</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_norepeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_norepeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_50_FPN_cosine_combined_all_norepeat.yaml">TFA w/ cos</a></td>
<td align="center">R-50</td>
<td align="center">22.7</td>
<td align="center">37.2</td>
<td align="center">23.9</td>
<td align="center">18.8</td>
<td align="center">27.7</td>
<td align="center">37.1</td>
<td align="center">15.4</td>
<td align="center">20.5</td>
<td align="center">28.4</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_norepeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_50_FPN_norepeat_cos/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_combined_all_norepeat.yaml">TFA w/ fc</a></td>
<td align="center">R-101</td>
<td align="center">23.9</td>
<td align="center">39.3</td>
<td align="center">25.3</td>
<td align="center">19.5</td>
<td align="center">29.5</td>
<td align="center">38.6</td>
<td align="center">16.2</td>
<td align="center">22.3</td>
<td align="center">28.9</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_norepeat_fc/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_norepeat_fc/metrics.json">metrics</a></td>
</tr>

<tr><td align="left"><a href="configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml">TFA w/ cos</a></td>
<td align="center">R-101</td>
<td align="center">24.3</td>
<td align="center">39.3</td>
<td align="center">25.8</td>
<td align="center">20.1</td>
<td align="center">30.2</td>
<td align="center">39.5</td>
<td align="center">18.1</td>
<td align="center">21.8</td>
<td align="center">29.8</td>
<td align="center"><a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_norepeat_cos/model_final.pth">model</a>&nbsp;|&nbsp;<a href="http://dl.yf.io/fs-det/models/lvis/R_101_FPN_norepeat_cos/metrics.json">metrics</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>
