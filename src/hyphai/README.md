## src/hyphai/

In this directory, you will find the source code for the project. The source code is written in Python using PyTorch framework. The source code is organized into the following directories:

  * `hyphai.py`: This file contains the abstract class `HyPhAIModule` which is the base class for all HyPhAI models.

  * `models.py`: This file contains the HyPhAI models.

  * `schemes.py` : Time schemes used in the project.

  * `equations.py` : Equations used as physical information in the HyPhAI models.

  * `unet_xception.py` : The U-Net Xception-style model used in in HyPhAI models.

  * `unet.py` : Standard U-Net model.

  * `utils.py`: This file contains functions used in the `examples` folder.

  * `metrics.py`: This file contains the metrics used for evaluation.

  * `dataset.py`: This file contains the torch DataSet class used in the project.

  * `trainers.py`: This file contains the training code used in the project. This file require to request the data from the EUMETSAT. 100 training samples are available at ![][https://zenodo.org/badge/DOI/10.5281/zenodo.10642094.svg].