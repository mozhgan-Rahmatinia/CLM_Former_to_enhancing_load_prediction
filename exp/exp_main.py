from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, CLM_Former, Transformer, Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'CLM_Former': CLM_Former,
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # ===============================
    # TRAIN FUNCTION (modified)
    # ===============================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # ✅ این حلقه باید داخل حلقه epoch باشد
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = outputs[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # 🔹 آزادسازی حافظه بعد از هر batch
                del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp
                torch.cuda.empty_cache()

            # ✅ validation و early stopping بعد از هر epoch
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch {epoch+1} | Train Loss: {np.average(train_loss):.6f} | Val Loss: {vali_loss:.6f} | Test Loss: {test_loss:.6f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # --- Save the best model manually ---
        best_model_path = os.path.join(path, 'best_model.pth')
        torch.save(self.model.state_dict(), best_model_path)
        print(f"✅ Best model saved at: {best_model_path}")

        # --- Count total trainable parameters ---
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        num_params = count_parameters(self.model)
        print(f"Total Trainable Parameters: {num_params / 1e6:.2f} M")

        return self.model



    # ===============================
    # TEST FUNCTION (modified)
    # ===============================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        model_path = os.path.join(self.args.checkpoints, setting, 'best_model.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded best model from {model_path}")
        else:
            print("⚠️ Best model not found. Using current weights.")

        self.model.eval()
        total_mae, total_mse, total_rmse, total_mape, total_mspe = 0, 0, 0, 0, 0
        count = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=self.device)

        start_time = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                torch.cuda.synchronize()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.output_attention:
                    outputs = outputs[0]
                torch.cuda.synchronize()

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds_np = outputs.detach().cpu().numpy()
                trues_np = batch_y.detach().cpu().numpy()

                mae, mse, rmse, mape, mspe = metric(preds_np, trues_np)
                total_mae += mae * len(preds_np)
                total_mse += mse * len(preds_np)
                total_rmse += rmse * len(preds_np)
                total_mape += mape * len(preds_np)
                total_mspe += mspe * len(preds_np)
                count += len(preds_np)

                if i % 20 == 0:
                    print(f"Batch {i}: processed")

                del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, preds_np, trues_np
                torch.cuda.empty_cache()

        avg_mae = total_mae / count
        avg_mse = total_mse / count
        avg_rmse = total_rmse / count
        avg_mape = total_mape / count
        avg_mspe = total_mspe / count

        end_time = time.time()
        total_time = end_time - start_time
        peak_mem = torch.cuda.max_memory_allocated(device=self.device) / (1024**2)

        print(f"✅ Test completed.")
        print(f"MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")
        print(f"Total Test Time: {total_time:.2f} s")
        print(f"Peak Memory: {peak_mem:.2f} MB")

        torch.cuda.empty_cache()
        return avg_mae, avg_mse, avg_rmse, avg_mape, avg_mspe, total_time, peak_mem

