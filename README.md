Code used for Adversarial training (with and without dreams) is contained in lightning_trainer.py.

All experiments have used this codebase with different parameters setup.
Single experiments can be run using run_lightning.py, batches of experiments can be started from run_all.
Validation afterwards requires the saved model locations to be set in run_all_validations.py.

Several notebooks can be found in the notebooks folder, which visualize or plot a number of things.

If interested, one can locally run this project by installing the required packages from curpackages.txt. 
For the notebooks a trained model is required, for which we recommend using GPU accelerated hardware although CPU usage should be supported as well.

If one would like to visualize and play with dream generation, the notebook ``dreamvis'' could be interesting.