Step-by-Step Usage
1. Create Data Files <br>
   Navigate to `data/` <br>
   Open and execute: `run_data_prep.ipynb`

2. Move to Main Folder
Change directory to `main/` for all following steps.

3. Training a Predictive Model
 * Train the model:`python train_model.py --device=cuda`
 * Test the predictive model: `python test_model.py`

4. Calculate Temporal Attribution:<br>
  * Run for each dataset (train, valid, test): `python temporal_attribution.py`
 
5. Evaluate Temporal Attribution <br>
* Train model with only top u% time points unmasked: <br>
     `python train_with_masking.py --device=cuda`     
* Test the model trained with partial sequence: <br>
     `python test_with_masking.py`

