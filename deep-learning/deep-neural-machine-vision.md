# Deep Neural Machine Vision

This page collects tools and notes on detection, recognition, and segmentation in machine vision.
Sections move from tools and super resolution into detection, recognition, and segmentation.

The same notes are in [Machine Vision annotation](../data/annotation-and-disagreement.md#machine-vision-annotation) and [Vision](../generative-ai/vision.md).

## TOOLS

This section points at image deduplication and Segment Anything as the vision tooling entry.

- GitHub - idealo/imagededup: 😎 Finding duplicate images made easy! [Image deduplication](https://github.com/idealo/imagededup)
2. Segment anything by Meta

## SUPER RESOLUTION

This section points at a state-of-the-art comparison after the tools above.

- "Zero Shot" Super-Resolution using Deep Internal Learning. [State of the art comparison](http://www.wisdom.weizmann.ac.il/~vision/zssr/)

## DETECTION

This section is object detection: the R-CNN family, YOLO, and related code after super resolution above.

The same notes are in [CONVOLUTIONAL NEURAL NET](convolutional-nets.md#convolutional-neural-net) and [Mix N Match](../generative-ai/mix-n-match.md).

<figure><img src="../.gitbook/assets/gimg-14a93cb242ce.png" alt=""><figcaption><p>Detection</p><p>Credit: <a href="https://lh5.googleusercontent.com/Efe-9nD1W6Hes040DI2Zgm2lzh0vnkYVTB95hnK1rmv3DYtfbPt9Bia0iVnSV49xJRs8JYLggj7KvIRGZDpbz4melmLvp0uLwQ-F6wtCjHYwRKjD4rw7DH8p90Gqo-P4DZNpW8fH">copied from the original hosted image</a>.</p></figcaption></figure>

1. [Review on DL technique applied to semantic segmentation](https://arxiv.org/pdf/1704.06857.pdf)
- A Gentle Introduction to Object Recognition With Deep Learning - MachineLearningMastery.com, by Jason Brownlee. [Mastery on obj detection](https://machinelearningmastery.com/object-recognition-with-deep-learning/)
- FAIR's research platform for object detection research, implementing popular algorithms like Mask R-CNN and RetinaNet. Fair [detectron](https://github.com/facebookresearch/Detectron)
- Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch. [Maskrcnn benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- Abstract page for arXiv paper 1703.06870: Mask R-CNN. [paper](https://arxiv.org/abs/1703.06870)
- A Simple and Versatile Framework for Object Detection and Instance Recognition - tusen-ai/simpledet. [Simpledet - obj detection and instance recognition](https://github.com/TuSimple/simpledet)
- OpenMMLab Detection Toolbox and Benchmark. [Mmdetection](https://github.com/open-mmlab/mmdetection)
- ResearchGate - Temporarily Unavailable. ResearchGate - Temporarily Unavailable. [Blind image separation](https://www.researchgate.net/publication/3938186_Blind_image_separation_through_kurtosis_maximization)
- Deep Learning for Image Segmentation: U-Net Architecture - Fritz ai. [UNET](https://heartbeat.fritz.ai/deep-learning-for-image-segmentation-u-net-architecture-ff17f6e4c1cf)
- The code for our newly accepted paper in Pattern Recognition 2020: "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection." - xuebinqin/U-2-Net. [U^2 Net - using a detection network for pencil drawing generation and segmentation](https://github.com/NathanUA/U-2-Net)
10. FastAI image segmentation

<figure><img src="../.gitbook/assets/gimg-6e15f36aa9d3.png" alt=""><figcaption><p>Detection</p><p>Credit: <a href="https://lh6.googleusercontent.com/0gWJVORnNeoeKD6j3fwo1HrA9W8SN2ZHUBkX8YdhLUomtniJ8tlattamydryookCJrL3Pu35a3xZUfOpkc3jXYBsm0gAkMZl5IxCg5nijzRSX80vwvethJRbWGK662LnMfLw4lcZ">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-ad4bdaa06cf7.png" alt=""><figcaption><p>Detection</p><p>Credit: <a href="https://lh5.googleusercontent.com/kn9eEm1IltsrjvpNUJsS9iZ0zgFynCyqA2kk4OCN9EjFRXKqeUrKlvv7UbfbvwPfQ-kz0fOn3kpUqnE3liGs71m9945BLBPmpeFtOdzCyp6FUhA-7_AEjvzYnaDTXUnz-JEsbWHS">copied from the original hosted image</a>.</p></figcaption></figure>

- Abstract page for arXiv paper 1506.02640: You Only Look Once: Unified, Real-Time Object Detection. [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- Abstract page for arXiv paper 1612.08242: YOLO9000: Better, Faster, Stronger. [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
- Abstract page for arXiv paper 1804.02767: YOLOv3: An Incremental Improvement. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- R-CNN: Regions with Convolutional Neural Network Features - rbgirshick/rcnn. . [R-CNN: Regions with Convolutional Neural Network Features, GitHub](https://github.com/rbgirshick/rcnn)
- GitHub - rbgirshick/fast-rcnn: Fast R-CNN. . [Fast R-CNN, GitHub](https://github.com/rbgirshick/fast-rcnn)
- Faster R-CNN (Python implementation) -- see https://github.com/ShaoqingRen/faster_rcnn for the official MATLAB version - rbgirshick/py-faster-rcnn. . [Faster R-CNN Python Code, GitHub](https://github.com/rbgirshick/py-faster-rcnn)
- YOLO: Real Time Object Detection. . [YOLO, GitHub](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)
- Abstract page for arXiv paper 1311.2524: Rich feature hierarchies for accurate object detection and semantic segmentation. [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
- Abstract page for arXiv paper 1406.4729: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
- Abstract page for arXiv paper 1504.08083: Fast R-CNN. [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- Abstract page for arXiv paper 1506.01497: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- Abstract page for arXiv paper 1703.06870: Mask R-CNN. [Mask R-CNN](https://arxiv.org/abs/1703.06870)
23. [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4), 2017.
- Object Recognition For Dummies Part 3. [Object Detection for Dummies Part 3: R-CNN Family](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)
- IE=edge"> <meta name="viewport" content="width=device-width, initial-scale=1"> <title>Object Detection Part 4</title> <meta name="description" content=""> <meta content="Lil'Log" property="og:site_name"> <meta content=. [Object Detection Part 4: Fast Detection Models](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)
- IKEA Assembly Dataset. IKEA Assembly Dataset. [Ikea ASM](https://ikeaasm.github.io/)

## RECOGNITION

This section points at image recognition that uses hashtags after detection above.

- Advancing state-of-the-art image recognition with deep learning on hashtags - Engineering at Meta. [Using image hashtags](https://engineering.fb.com/ml-applications/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/)

## Segmentation

This section points at a ViT segmentation page after recognition above.

The same notes are in [CONVOLUTIONAL NEURAL NET](convolutional-nets.md#convolutional-neural-net) and [Mix N Match](../generative-ai/mix-n-match.md).

- Project page for 'Deep ViT Features as Dense Visual Descriptors.'. [Vit](https://dino-vit-features.github.io/)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Segment anything by Meta. This address no longer opens: https://segment-anything.com/demo
- FastAI image segmentation. This address no longer opens: https://gilberttanner.com/blog/fastai-image-segmentation
