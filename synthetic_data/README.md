Step-by-Step Usage
1. Generate Raw Data <br>
   Navigate to `generating_raw_data/` <br>
   Open and execute: `run_generator.ipynb`

2. Create Data Files <br>
   Navigate to `data/` <br>
   Open and execute: `run_data_prep.ipynb`

3. Move to Main Folder
Change directory to `main/` for all following steps.

4. Training a Predictive Model
 * Train the model:`python train_model.py --device=cuda`
 * Test the predictive model: `python test_model.py`

5. Calculate Temporal Attribution:<br>
  * Run for each dataset (train, valid, test): `python temporal_attribution.py`
 
6. Evaluate Temporal Attribution <br>
* Train model with only top u% time points unmasked: <br>
     `python train_with_masking.py --device=cuda`     
* Test the model trained with partial sequence: <br>
     `python test_with_masking.py`

