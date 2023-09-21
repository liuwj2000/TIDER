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

•	`--lambda_trend`: control the loss function of trend matrix


For Guangzhou data: It is a 214*61*144 tensor, we can first transfer it into a 214*8784 matrix, then use the last 500 columns as the processed matrix

For Westminster data: a csdn blog is close to our processed: https://blog.csdn.net/qq_40206371/article/details/128932640


# AdaTIDER

nearly same as TIDER. The differences lies in:

• no hyperparameter 'seasonality'

• lambda_spatial: control the Laplacian regularization term for matrix U

• topk_freq: the number of frequencies selected in multi-periods seasonality matrix.


