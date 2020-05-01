# Vision Final
This repo is the work done as part of the final project for ECE 471. 

# Running This Project
We found that you needed quite a powerful machine to run this project effectively. Through trial and error on Google Cloud Compute, we ended up at 10 vCPUs, 30 GB memory as recommended by the engine itself. You will need the following:
* Python 2.7
* Pip
* Node.js 
* NPM

We have included the version of the model we have trained as well as all files we created and used in order to let you train the model from scratch if you so choose. 

## Getting the Data
The following steps can be run to get the data required:
1. Download the dataset by running `./downloadDataset.sh`
2. Get a subset of the Labeled Faces in the Wild dataset by running `npm install && node parse.js`
3. Run `rm -rf lwf && mv lfw10/ lfw/` to replace the original LFW dataset with the new subsetted version
4. Split the data into test and train sets by running `python makeTrainCSV.py`

The output should be two files: `test1.csv` and `train1.csv`.

## Training the Model
The following steps can be used to train the model
1. Create a directory called `saved_model`
2. Run `pip install -r requirements.txt`
3. Run `python runner.py`

On the GCP instance we were using this took at least a day and a half, so depending on how strong of a machine you're running this on it will take either more or less time. Once the model is done training, the output will be written to the `saved_model` dir which should now contain `encoding_network_arch.json`, `encoding_network_weights.h5`, `siamese_model_weights.h5`, and `siamese_network_arch.json`.

## Evaluating the Model
The following steps can be used to evaluate the model
1. Run `evaluateSiamese.py`

The evaluation code will print an accuracy percentage to the terminal once it finishes running.

