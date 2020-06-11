# bcb-crater-detection
Impact craters are the most dominant landmarks on many celestial bodies and crater counts are the only available tool for measuring remotely the relative ages of geologic formations on planets.  Automatic detection of craters can reveal information about the past geological processes.  However, crater detection is much more challenging than typical object detection like face detection due to lack of common features like size and shape among craters and, varying nature of the surrounding planetary surface. 

This project proposes a new crater detection framework named BCB-Crater-Detection that learns bidirectional context-based features from both crater and non-crater ends.  This framework utilizes both craters and its surrounding features using deep convolutional classification and segmentation models to identify efficiently sub-kilometer craters in high-resolution panchromatic images.

The BCB-Crater-Detection framework includes non-crater pixel-level segmentation, crater level classifier, and refinement steps.  A segmentation model is designed to detect non-crater areas at pixel-level.  A crater classifier designed using deep convolutional filters and combined them to learn robust discriminative features.  The crater classifier model is trained by progressively re-sizing and ensemble learning methods.  Then, sliding window and pyramid techniques are applied to detect craters and generates final predictions.  A crater score measure is defined to combine the outputs of non-crater and crater detection steps and produce crater predictions. 


