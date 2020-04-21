import torch
import torch.nn.functional as F


class Fd(torch.nn.Module):
    """
    Simple fully connected network.
    """

    def __init__(self, in_dim, H, out_dim, fin_act=None):
        super(Fd, self).__init__()
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)
        self.fin_act = fin_act

    def forward(self, x):
#         x = self.dropout(F.leaky_relu(self.fc1(x), .1))
#         x = self.fc2(x)
        x = F.leaky_relu(self.fc1(x), .1)
        x = self.fc2(x)
        return self.fin_act(x) if self.fin_act else x


class Fx(torch.nn.Module):

    def __init__(self, in_dim, H1, H2, out_dim):
        super(Fx, self).__init__()
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(in_dim, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, out_dim)

    def forward(self, x):
#         x = self.dropout(F.leaky_relu(self.fc1(x), 0.1))
#         x = self.dropout(F.leaky_relu(self.fc2(x), 0.1))
#         x = self.dropout(F.leaky_relu(self.fc3(x), 0.1))
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        return x


class Fe(torch.nn.Module):

    def __init__(self, in_dim, H, out_dim):
        super(Fe, self).__init__()
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)

    def forward(self, x):
#         x = self.dropout(F.leaky_relu(self.fc1(x), 0.1))
#         x = self.dropout(F.leaky_relu(self.fc2(x), 0.1))
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        return x


class C2AE(torch.nn.Module):

    def __init__(self, Fx, Fe, Fd, alpha=.5, emb_lambda=.5, latent_dim=6,
                 device=None):
        super(C2AE, self).__init__()
        # Define main network components.
        # Encodes x into latent space. X ~ z
        self.Fx = Fx
        # Encodes y into latent space. Y ~ z
        self.Fe = Fe
        # Decodes latent space into Y. z ~ Y
        self.Fd = Fd

        # Hyperparam used to set tradeoff between latent loss, and corr loss.
        self.alpha = alpha
        # Lagrange to use in embedding loss.
        self.emb_lambda = emb_lambda
        self.latent_I = torch.eye(latent_dim).to(device)

    def forward(self, x, y=None):
        """
        Forward pass of C2AE model.

        Training:
            Runs feature vector x through Fx, then encodes y through Fe and
            computes latent loss (MSE between feature maps). Then z = Fe(y) is
            sent through decoder network in which it tries to satisfy
            correlation equation.
            TODO: ADD MORE DESCRIPTION HERE

        Testing:
            Simply runs feature vector x through autoencoder. Fd(Fx(x))
            This will result in a logits vec of multilabel preds.
        """
        if self.training:
            # Calculate feature, and label latent representations.
            fx_x = self.Fx(x)
            fe_y = self.Fe(y)
            # Calculate decoded latent representation.
            fd_z = self.Fd(fe_y)
            return fx_x, fe_y, fd_z
        else:
            # If evaluating just send through encoder and decoder.
            return self.predict(x)

    def _predict(self, y):
        """This method predicts with the y encoded latent space.
        """
        return self.Fd(self.Fe(y))

    def predict(self, x):
        """This method predicts with the x encoded latent space.
        """
        return self.Fd(self.Fx(x))

    def corr_loss(self, preds, y):
        """This method compares the predicted probabilitie class distribution
        from the decoder, with the true y labels.

        Does this by [BLAH]
        """
        # Generate masks for [0,1] elements.
        ones = (y == 1)
        zeros = (y == 0)
        # Use broadcasting to apply logical and between mask arrays.
        # This will only indicate locations where both masks are 1.
        # THis corresponds to set we are enumerating in eq (3) in Yah et al.
        ix_matrix = ones[:, :, None] & zeros[:, None, :]
        # Use same broadcasting logic to generate exponetial differences.
        # This like the above broadcast will do so between all pairs of points
        # for every datapoint.
        diff_matrix = torch.exp(-(preds[:, :, None] - preds[:, None, :]))
        # This will sum all contributes to loss for each datapoint.
        losses = torch.flatten(diff_matrix*ix_matrix, start_dim=1).sum(dim=1)
        # Normalize each loss
#         print(torch.isnan(ones.sum(dim=1)).sum(), torch.isnan(zeros.sum(dim=1)).sum(), 'card')
#         print((ones.sum(dim=1)*zeros.sum(dim=1) == 0).sum(), 'zero_sum')
        losses /= (ones.sum(dim=1)*zeros.sum(dim=1) + 1e-4)
        # Replace inf, and nans with 0.
        losses[losses == float('Inf')] = 0
        losses[torch.isnan(losses)] = 0
        # Combine all losses to retrieve final loss.
        return losses.sum()

    def latent_loss(self, fx_x, fe_y):
        """
        Loss between latent space generated from fx, and fe.

        ||Fx(x) - Fe(y)||^2 s.t. FxFx^2 = FyFy^2 = I

        This is seen in equation (2), and implemention details seen in 
        decomposed version of loss.

        First version contains decomposition of loss function, making use of 
        lagrange multiplier to account for constraint.

        Second version just calculates the mean squared error.
        """
        # ********** Version 1: Implemented as suggested in Yeh et al. # **********
        # Initial condition.
        c1 = fx_x - fe_y
        # Here to help hold constraint of FxFx^2 = FyFy^2 = I.
        c2 = fx_x.T@fx_x - self.latent_I
        c3 = fe_y.T@fe_y - self.latent_I
        # Combine loss components as specified in Yah et al.
        latent_loss = torch.trace(
            c1@c1.T) + self.emb_lambda*torch.trace(c2@c2.T + c3@c3.T)
        # ********** Version 2: Ignore constraint **********
        #latent_loss = torch.mean((fx_x - fe_y)**2)
        return latent_loss

    def losses(self, fx_x, fe_y, fd_z, y):
        """This method calculates the main loss functions required
        when composing the loss function.
        """
        l_loss = self.latent_loss(fx_x, fe_y)
        c_loss = self.corr_loss(fd_z, y)
        return l_loss, c_loss


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_cls, path, *args, **kwargs):
    model = model_cls(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def eval_metrics(mod, metrics, datasets, device):
    res_dict = {}
    for ix, dataset in enumerate(datasets):
        mod.eval()
        x = dataset.tensors[0].to(device)
        # Make predictions.
        preds = mod(x)
        # Convert them to binary multilabels.
        y_pred = torch.round(preds).cpu().detach().numpy()
        y_true = dataset.tensors[1].cpu().detach().numpy()
        # Calculate metric.
        res_dict[f'dataset_{ix}'] = {metric.__name__: metric(
            y_true, y_pred) for metric in metrics}
    return res_dict
