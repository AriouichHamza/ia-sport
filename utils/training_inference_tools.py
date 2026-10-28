import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import utils
from sklearn.model_selection import TimeSeriesSplit
from typing import Any
import numpy as np

def print_env():
    if torch.cuda.is_available():
        print("gpu is available")
    else:
        print("gpu is not available")

def get_optimizer(model_params, args):
    return torch.optim.Adam(model_params, lr=args.lr)

def get_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.0001, patience=50, verbose=True)

def train_model(args, model, lst_losses,train_dataloader,val_dataloader,metrics):
    best_metric = float('inf')  # Initialize with positive infinity for loss
    train_epochs_mae_losses = []
    train_epochs_mse_losses = []
    val_epochs_mae_losses = []
    val_epochs_mse_losses = []
    label_len = 48
    optimizer = get_optimizer(model.parameters(),args)
    #lr_scheduler = get_scheduler(optimizer)
    for epoch in range(args.n_epochs):
        model.train()
        epoch_mse_loss = 0
        epoch_mae_loss = 0
        for batch, item in enumerate(train_dataloader):
            x, y = item
            # print("x ", x.shape)
            # print("y ", y.shape)
            n_features = x.shape[2] if x.ndim == 3 else 1
            x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
            #print_env() #print env
            x = x.to(args.device)  # for GPU
            y = y.to(args.device)  # for GPU
            if args.selected_model == 'VanillaTransformer':
                dec_inp = y[:, -label_len:, :]  # get last (48,num_series) lines of each batch of the forecast data
                dec_inp_zeros = torch.zeros_like(y[:, -y.shape[1]:, :]).float()  # mask of zeros of forecasting shape
                dec_inp = torch.cat([dec_inp, dec_inp_zeros], dim=1).float().to(args.device) #[B, time_dim+48, num_series]]
                output = model(x, dec_inp)
            else:
                output = model(x)  # generate preds
                
            optimizer.zero_grad()  # zero out gradients
            #if args.selected_model == 'N_BEATS':
            #    output = pred[1]
            mse_loss = lst_losses[0](output, y)  # compute mse loss
            mae_loss = metrics.mae(output.detach().cpu().numpy(), y.cpu().numpy())  # compute mae loss

            mse_loss.backward()  # calculate mse gradients by backprop
            optimizer.step()  # update mse gradients

            epoch_mse_loss += mse_loss.item()  # sum up losses
            epoch_mae_loss += mae_loss  # sum up losses

        avg_mse_training_loss = epoch_mse_loss / len(train_dataloader)
        avg_mae_training_loss = epoch_mae_loss / len(train_dataloader)
        train_epochs_mse_losses.append(float(avg_mse_training_loss))  #
        train_epochs_mae_losses.append(float(avg_mae_training_loss))  #


        # Step 6: Validation loop
        model.eval()  # Set the model in evaluation mode
        val_epoch_mse_loss = 0
        val_epoch_mae_loss = 0

        with torch.no_grad():
            for batch, item in enumerate(val_dataloader):
                x, y = item
                n_features = x.shape[2] if x.ndim == 3 else 1
                x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
                #print_env()  # print env
                x = x.to(args.device)  # for GPU
                y = y.to(args.device)  # for GPU

                if args.selected_model == 'VanillaTransformer':
                    dec_inp = y[:, -label_len:, :]  # get last (48,num_series) lines of each batch of the forecast data
                    dec_inp_zeros = torch.zeros_like(y[:, -y.shape[1]:, :]).float()  # mask of zeros of forecasting shape
                    dec_inp = torch.cat([dec_inp, dec_inp_zeros], dim=1).float().to(args.device) #[B, time_dim+48, num_series]]
                    output = model(x, dec_inp)
                else:
                    output = model(x)  # generate preds
                # if args.selected_model == 'N_BEATS':
                #     output = output[1]
                mse_loss = lst_losses[0](output, y)  # mse loss
                mae_loss = lst_losses[1](output, y)  # mae loss
                val_epoch_mse_loss += mse_loss.item()
                val_epoch_mae_loss += metrics.mae(output.detach().cpu().numpy(), y.cpu().numpy())  # 4 GPU

        avg_mse_val_loss = val_epoch_mse_loss / len(val_dataloader)
        avg_mae_val_loss = val_epoch_mae_loss / len(val_dataloader)

        # lr_scheduler.step(avg_mae_val_loss)
        val_epochs_mse_losses.append(float(avg_mse_val_loss))
        val_epochs_mae_losses.append(float(avg_mae_val_loss)) 
        print(
            f'Epoch [{epoch + 1}/{args.n_epochs}] Epoch {epoch + 1}: mse:- {avg_mse_training_loss:.5f} - mae:- {avg_mae_training_loss:.5f} - val_mse: {avg_mse_val_loss:.5f} - val_mae: {avg_mae_val_loss:.5f}')
        if args.save_optimal_model is True:
            if (avg_mae_val_loss < best_metric):  # save best model
                print(
                    "Epoch [{}/{}] : val_loss improved from {:.5f} to {:.5f}, saving model to Checkpoint_file".format(
                        epoch + 1, args.n_epochs, best_metric, avg_mae_val_loss))
                best_metric = avg_mae_val_loss

                utils.save_model(model, epoch + 1, optimizer, args.save_model_PATH) #save optimal model

    return train_epochs_mae_losses, val_epochs_mae_losses, optimizer



def test_model(args, model, mae_loss,test_dataloader,optimizer,metrics):
    #load best model during training
    if args.save_optimal_model:
        print("uploading optimal for testing...")
        utils.load_model(model, args.save_model_PATH, optimizer)  # load sub_model
    model.eval()
    test_loss = 0
    r2_score = 0
    flattened_pred = []
    flattened_y = []
    label_len = 48
    with torch.no_grad():
        for batch, item in enumerate(test_dataloader):  # 8 batches of 64
            x, y = item
            if (batch < 1):
                print("--P_h {} - F_h {}--".format(x.shape[1], y.shape[1]))
            n_features = x.shape[2] if x.ndim == 3 else 1
            x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
            print_env()  # print env
            x = x.to(args.device)
            y = y.to(args.device)
            if args.selected_model == 'VanillaTransformer':
                dec_inp = y[:, -label_len:, :]  # get last (48,num_series) lines of each batch of the forecast data
                dec_inp_zeros = torch.zeros_like(y[:, -y.shape[1]:, :]).float()  # mask of zeros of forecasting shape
                dec_inp = torch.cat([dec_inp, dec_inp_zeros], dim=1).float().to(args.device) #[B, time_dim+48, num_series]]
                output = model(x, dec_inp)
            else:            
                output = model(x)  # generate preds
            #elif args.selected_model == 'N_BEATS':
            #    output = pred[1]
            #else:
            #    output = model(x)
            output = output.detach().cpu()
            y = y.detach().cpu()
            flattened_pred.append(output)
            flattened_y.append(y)
            loss = mae_loss(output, y)
            test_loss += loss.item()
            r2_score += metrics.r2_avr(output.permute(2,0,1), y.permute(2,0,1))
            # print(f'Testing - Batch Loss: {loss.item()}')
            # print(f'Testing - Batch R2 Loss: {metrics.r2_score(pred,y)}')
    print(
        f'Testing - Average Loss: {test_loss / len(test_dataloader)} - Average R2 Loss: {r2_score / len(test_dataloader)}')
    flattened_pred = torch.cat(flattened_pred, dim=0)  # concat all pred batches
    flattened_y = torch.cat(flattened_y, dim=0)  # concat all label batches

    return flattened_pred.numpy() , flattened_y.numpy() 



def train_model_with_cross_val(model:Any, X:np.array, y:np.array, n_splits:int, metrics:Any, verbose=True)->list:
    """
    Perform time series cross-validation using sklearn's TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # print("here {} - {}".format(train_idx,test_idx))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metrics.rmse(y_test, y_pred)
        scores.append(score)

        if verbose:
            print(f"Fold {i+1}/{n_splits}: RMSE = {score:.4f}")

    if verbose:
        print(f"\nAverage RMSE: {np.mean(scores):.4f}")

    return scores