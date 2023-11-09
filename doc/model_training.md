# Model Training

Before proceeding with model training, ensure that the data is preprocessed, and the required dependencies are installed as per the initialization documentation.

## Data Import for Model Training

To begin model training, import the labeled data file from the labeling tool and place it in the `data` folder of your project. Ensure that the data file is named appropriately for reference in the subsequent steps.

## Notebook Execution

1. Open the `model_training.ipynb` notebook.

2. In the second Python cell of the notebook, specify which component you want to train. Modify the cell accordingly to indicate whether you are training the NER (Named Entity Recognition) component or the REL (Relation Extraction) component. For example:

   ```python
   component = 1 # or 2
    ```

This step is crucial for ensuring that the correct model is trained based on your requirements.
1. Execute the notebook cells to initiate the training process. The notebook will train models for the specified component(s).
2. At the end of the notebook, there will be an example demonstrating the combination of NER and REL models to analyze a sample text.

The trained models will be saved in the appropriate model folder under `training_{component_number}`. These models will later be utilized in the output notebook to generate a knowledge graph.