# TIDER
Code for paper titled 'Multivariate Time-series Imputation with Disentangled Temporal Representations'.

Code is written in PyTorch v1.9.0+cu111. Python version is Python 3.6.9.  

## To run the model

Simply 'python3 TIDER.py' can conduct the training, validation and testing process.

##Detailed explanations of hyperparameters:
•	`--save_path`: address to save the optimal model parameters.

•	`--datadir`: address of time-series data.

•	`--device`: cpu / gpu device.

•	`--valid`: proportion of validation data.

•	`--drop_rate`: data removing rate

•	`--eta`: hyperparameter for L2 regularization

•	`--n_test`: testing set temporal length

•	`--num_epochs`: number of epochs

•	`--batch_size`: batch size

•	`--dim_size`: dimension of feature matrix

•	`--lag_list`: W of bias matrix

•	`--lambda_ar`: weight for bias matrix constraint function

•	`--lambda_trend`: weight for trend matrix constraint function

•	`--bias_dimension`: dimension for bias feature matrix

•	`--season_num`: K

•	`--seasonality`: seasonality of time-series

•	`--learning_rate`: learning rate

•	`--lambda_trend`: learning rate



