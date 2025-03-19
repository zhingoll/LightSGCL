# LightSGCL

A novel Graph Contrast Learning method combining Single Feature Fusion Graph Contrast Learning(S-GCL) and Column Vector Orthogonal Noise(CVON) for recommendation.

## Key Components

1. **LightSGCL Implementation**: The core implementation of the LightSGCL can be found in the `LightSGCL.py` file. This file contains the main architecture and algorithmic logic of the model.
2. **Column Vector Orthogonal Noise**: The CVON feature augmentation method is implemented in the `construct_noise_matrix` method within `loss_torch.py`.

### Configuration and Execution

To run the LightSGCL model within the [SELFRec framework](https://github.com/Coder-Yu/SELFRec), please follow these steps:

1. **Environment Setup**: Ensure that your runtime environment meets the dependencies required by SELFRec.
2. **Model Configuration**: Relevant hyperparameters for the model can be found and adjusted in the `LightSGCL.conf` file.
3. **Run the Model**: Once the model is configured according to the guidance in the SELFRec documentation, it is ready to be executed.

4. ### Contact Us

If you encounter any issues while using the model, or if you would like to discuss technical details with me, please feel free to contact me through the issues.
