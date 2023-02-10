Code refactoring are undergoing for better understanding.

# Smartee project: 3D Teeth Reconstruction from Five Orthodontic Photos

## Overview
We develop a template-based framework leveraging the prior shape knowledge of human teeth to reconstruct digital 3D models of upper and lower teeth from the typical five orthodontic intra-oral photographs. The framework consists of three phases: parameterization of the arrangement and shape of teeth through statistical shape modelling, U-net based teeth boundary extraction in intra-oral images, and 3D teeth reconstruction based on the prior parametric teeth model.
<p align="center">
    <img src=".\demo\assets\teeth_reconstruction_framework.png" alt="teeth reconstruction framework" width="600"/>
<\p>

## Reconstruction result

Maxillary view |  Mandibular view | Left view | Right view | Anterior view
:----:|:----:|:----:|:----:|:----:
<img src=".\seg\valid\image\1-0.png" alt="orthodontic photo: maxillary view" width="100"/>|<img src=".\seg\valid\image\1-1.png" alt="orthodontic photo: mandibular view" width="100"/>|<img src=".\seg\valid\image\1-2.png" alt="orthodontic photo: left view" width="100"/>|<img src=".\seg\valid\image\1-3.png" alt="orthodontic photo: right view" width="100"/>|<img src=".\seg\valid\image\1-4.png" alt="orthodontic photo: anterior view" width="100"/>
<img src=".\demo\visualization\mesh-tag=1-PHOTO.UPPER.png" alt="reconstructed teeth: maxillary view" width="100"/>|<img src=".\demo\visualization\mesh-tag=1-PHOTO.LOWER.png" alt="reconstructed teeth: mandibular view" width="100"/>|<img src=".\demo\visualization\mesh-tag=1-PHOTO.LEFT.png" alt="reconstructed teeth: left view" width="100"/>|<img src=".\demo\visualization\mesh-tag=1-PHOTO.RIGHT.png" alt="reconstructed teeth: right view" width="100"/>|<img src=".\demo\visualization\mesh-tag=1-PHOTO.FRONTAL.png" alt="reconstructed teeth: anterior view" width="100"/>
<img src=".\demo\visualization\overlay-tag=1-PHOTO.UPPER.png" alt="projection: maxillary view" width="100"/>|<img src=".\demo\visualization\overlay-tag=1-PHOTO.LOWER.png" alt="projection teeth: mandibular view" width="100"/>|<img src=".\demo\visualization\overlay-tag=1-PHOTO.LEFT.png" alt="projection teeth: left view" width="100"/>|<img src=".\demo\visualization\overlay-tag=1-PHOTO.RIGHT.png" alt="projection teeth: right view" width="100"/>|<img src=".\demo\visualization\overlay-tag=1-PHOTO.FRONTAL.png" alt="projection teeth: anterior view" width="100"/>
