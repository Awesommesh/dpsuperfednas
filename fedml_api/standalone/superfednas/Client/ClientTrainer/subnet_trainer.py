from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
from fedml_api.standalone.superfednas.elastic_nn.TCN.word_cnn.utils import (
    get_batch,
)
import numpy as np
import torch
import logging
from torch import nn
import torch.nn.functional as F
from .dp_utils import compute_epsilon
from opacus import PrivacyEngine

class SubnetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(SubnetTrainer, self).__init__(model, device, args, teacher_model)
        self.test_model = model
        self.alpha = args.feddyn_alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def train(self, lr, local_ep, **kwargs):
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

        self.client_model.to(self.device)
        self.client_model.train()
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.running_mean.requires_grad = False
                    m.running_var.requires_grad = False
                    with torch.no_grad():
                        m.weight.fill_(1)
                        m.bias.fill_(0)
                        m.running_mean.fill_(0)
                        m.running_var.fill_(1)

        # train and update
        criterion = nn.CrossEntropyLoss().to(self.device)
        cur_wd = self.args.wd
        if (
            self.client_model.is_max_net(self.client_model.model_config)
            and self.args.largest_subnet_wd
        ):
            cur_wd = self.args.largest_subnet_wd
        if self.args.mod_wd_dyn:
            cur_wd += self.alpha

        model_params = list(filter(lambda p: p.requires_grad, self.client_model.model.parameters()))
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=cur_wd,)
        else:
            optimizer = torch.optim.Adam(
                model_params, lr=lr, weight_decay=cur_wd, amsgrad=True,
            )

            
        if self.args.use_opacus_dp:
            max_grad_norm = self.args.dp_clip_norm
            noise_multiplier = self.args.dp_noise_multiplier
            delta = self.args.dp_delta
            sample_rate = self.args.batch_size / len(self.local_training_data.dataset)
            
            privacy_engine = PrivacyEngine()
            self.client_model, optimizer, self.local_training_data = privacy_engine.make_private(
                module=self.client_model,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                optimizer=optimizer,
                data_loader=self.local_training_data
            )
        
        # dp-sgd args
        if self.args.use_dp:
            max_grad_norm = self.args.dp_clip_norm
            noise_multiplier = self.args.dp_noise_multiplier
            delta = self.args.dp_delta
            batch_size=64
        
            sample_rate = self.args.batch_size / len(self.local_training_data.dataset)
            steps = 0
            orders = [1 + x / 10.0 for x in range(1, 100)]
            
        # print(f"Number of batches: {len(self.local_training_data)}")
        epoch_loss = []
        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = []
            # Opacus test implementation
            if self.args.use_opacus_dp:
                for batch_idx, (x, labels) in enumerate(self.local_training_data):
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    log_probs = self.client_model.forward(x)
                    if self.args.model == "darts":
                        log_probs = log_probs[0]
                    if self.args.kd_ratio > 0:
                        with torch.no_grad():
                            soft_logits = self.teacher_model.forward(x).detach()
                            soft_label = F.softmax(soft_logits, dim=1)
                    if self.args.kd_ratio == 0:
                        loss = criterion(log_probs, labels)
                    else:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    batch_loss.append(loss.item())
            # dp-sgd impl. w/ per-sample
            elif self.args.use_dp:
                for batch_idx, (x, labels) in enumerate(self.local_training_data):
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    batch_size = x.size(0)
                    
                    aggregated_grads = [torch.zeros_like(param, device=self.device) for param in model_params]
                    
                    per_sample_grads = [
                        torch.zeros((batch_size, *param.shape), device=param.device)
                        for param in model_params
                    ]

                    for i in range(x.size(0)):
                        optimizer.zero_grad()

                        xi = x[i].unsqueeze(0)
                        label_i = labels[i].unsqueeze(0)
                        output_i = self.client_model.forward(xi)
                        if self.args.model == "darts":
                            output_i = output_i[0]
                        loss_i = criterion(output_i, label_i)

                        loss_i.backward()
                            
                        for idx, param in enumerate(model_params):
                            if param.grad is not None:
                                per_sample_grad = param.grad.detach().clone()
                                grad_norm = per_sample_grad.norm(2)
                                clip_coeff = max_grad_norm / (grad_norm + 1e-6)
                                clip_coeff = min(clip_coeff, 1.0)
                                per_sample_grad.mul_(clip_coeff)
                                aggregated_grads[idx] += per_sample_grad
                    
                    ########## OLD IMPLEMENTATION #################
                    #             per_sample_grads[idx][i] = param.grad.detach().clone()
                    # # print(f"Per sample grads: {per_sample_grads}")
                    # grad_norms = torch.zeros(x.size(0), device=self.device)
                    # for grad in per_sample_grads:
                    #     # grad is 64 x 3 x 3 x 3
                    #     temp = grad.view(x.size(0), -1).norm(2, dim=1) ** 2
                    #     # print(f"Here is a shape {temp.shape}")
                    #     grad_norms += grad.view(x.size(0), -1).norm(2, dim=1) ** 2
                    # grad_norms = grad_norms.sqrt()

                    # clip_coeffs = (max_grad_norm / (grad_norms + 1e-6)).clamp(max=1.0)
                    # # print(f"clip_coeffs: {clip_coeffs}")
                    
                    # # good up to here
                    # # theres an issue with .mul here
                    # for idx in range(len(per_sample_grads)):
                    #     # print(f"per_sample_grads[idx]: {per_sample_grads[idx].size()}")
                    #     # print(f"clip_coeffs: {clip_coeffs.view(-1, 1).size()}")
                    #     new_shape = [1] * len(per_sample_grads[idx].shape)
                    #     new_shape[0] = -1
                    #     per_sample_grads[idx] = per_sample_grads[idx].mul(clip_coeffs.view(*new_shape))
                    # # print(f"Per_sample_grads: {len(per_sample_grads)}")
                    # # print(f"Per_sample_grads[0]: {per_sample_grads[0].size()}")

                    # aggregated_grads = []
                    # for grad in per_sample_grads:
                    #     # print(f'grad: {grad.size()}')
                    #     aggregated_grad = grad.sum(dim=0)
                    #     # print(f'agg_grad: {aggregated_grad.size()}')
                    #     aggregated_grads.append(aggregated_grad)
                    #     # print(f"grad: {grad.size()}")
                    # print(len(aggregated_grads))
                    
                    ########################################
                    
                    for idx, param in enumerate(model_params):
                        noise = torch.normal(
                            mean=0.0,
                            std=noise_multiplier * max_grad_norm,
                            size=aggregated_grads[idx].shape,
                            device=self.device,
                        )
                        aggregated_grads[idx].add_(noise)
                        param.grad = aggregated_grads[idx] / batch_size

                    optimizer.step()
                    
                    with torch.no_grad():
                        outputs = self.client_model.forward(x)
                        if self.args.model == "darts":
                            outputs = outputs[0]
                        loss = criterion(outputs, labels)
                        batch_loss.append(loss.item())
                        steps+=1
            elif self.args.dataset == 'ptb':
                for batch_idx, i in enumerate(range(0, self.local_training_data.size(1) - 1, self.args.validseqlen)):
                    if i + self.args.seq_len - self.args.validseqlen >= self.local_training_data.size(1) - 1:
                        continue
                    data, targets = get_batch(self.local_training_data, i, self.args)
                    self.client_model.zero_grad()
                    optimizer.zero_grad()
                    output = self.client_model.forward(data)

                    eff_history = self.args.seq_len - self.args.validseqlen
                    if eff_history < 0:
                        raise ValueError("Valid sequence length must be smaller than sequence length!")
                    final_target = targets[:, eff_history:].contiguous().view(-1)
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_words)
                    loss = criterion(final_output, final_target)
                    loss.backward()
                    if self.args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), self.args.max_norm)
                    optimizer.step()
                    batch_loss.append(loss.item())
            elif self.args.dataset == "shakespeare":
                for batch_idx, (data, targets) in enumerate(self.local_training_data):
                    #data, targets = data.to(self.device), targets.to(self.device)
                    self.client_model.zero_grad()
                    optimizer.zero_grad()
                    output = self.client_model.forward(data)

                    eff_history = data.size(1)-1
                    if eff_history < 0:
                        raise ValueError("Valid sequence length must be smaller than sequence length")
                    final_target = targets[:, eff_history:].contiguous().view(-1)
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_chars)
                    loss = criterion(final_output, final_target)
                    loss.backward()
                    if self.args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), self.args.max_norm)
                    optimizer.step()
                    cur_batch_loss = loss.item()
                    batch_loss.append(cur_batch_loss)
            else:
                for batch_idx, (x, labels) in enumerate(self.local_training_data):
                    x, labels = x.to(self.device), labels.to(self.device)
                    self.client_model.zero_grad()
                    log_probs = self.client_model.forward(x)
                    if self.args.model == "darts":
                        log_probs = log_probs[0]
                    if self.args.kd_ratio > 0:
                        with torch.no_grad():
                            soft_logits = self.teacher_model.forward(x).detach()
                            soft_label = F.softmax(soft_logits, dim=1)
                    if self.args.kd_ratio == 0:
                        loss = criterion(log_probs, labels)
                    else:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    loss.backward()

                    # to avoid nan loss
                    torch.nn.utils.clip_grad_norm_(
                        self.client_model.parameters(), self.args.max_norm
                    )

                    optimizer.step()

                    batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if self.args.use_dp:
                epsilon = compute_epsilon(steps, sample_rate, noise_multiplier, delta)
            if self.args.use_opacus_dp:
                epsilon, best_alpha = privacy_engine.get_privacy_spent(delta)
            if self.args.verbose:
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss),
                    )
                )
                if self.args.use_dp:
                    logging.info(f"Privacy budget after {steps} steps: ε = {epsilon:.2f}, δ = {delta}")
                if self.args.use_opacus_dp:
                    logging.info(
                        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
                    )
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert (
                            m.weight.equal(torch.ones_like(m.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            m.bias.equal(torch.zeros_like(m.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            m.running_mean.equal(torch.zeros_like(m.running_mean)),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            m.running_var.equal(torch.ones_like(m.running_var)),
                            "BN running var param not all 1s",
                        )
        if self.args.use_opacus_dp:
            privacy_engine.detach()
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model

        model.to(self.device)
        model.eval()
        if not args.use_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert (
                            m.weight.equal(torch.ones_like(m.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            m.bias.equal(torch.zeros_like(m.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            m.running_mean.equal(torch.zeros_like(m.running_mean)),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            m.running_var.equal(torch.ones_like(m.running_var)),
                            "BN running var param not all 1s",
                        )

        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            if self.args.dataset == 'ptb':
                metrics = {"test_total": 0, "test_ppl": 0}
                total_loss = 0
                processed_data_size = 0
                for i in range(0, dataset.size(1) - 1, args.validseqlen):
                    if i + args.seq_len - args.validseqlen >= dataset.size(1) - 1:
                        continue
                    data, targets = get_batch(dataset, i, args)
                    output = model.forward(data)

                    # Discard the effective history, just like in training
                    eff_history = args.seq_len - args.validseqlen
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_words)
                    final_target = targets[:, eff_history:].contiguous().view(-1)

                    loss = criterion(final_output, final_target)

                    # Note that we don't add TAR loss here
                    total_loss += (data.size(1) - eff_history) * loss.item()
                    processed_data_size += data.size(1) - eff_history
                metrics["test_loss"] = float(total_loss) / processed_data_size
                metrics["test_ppl"] = np.exp(metrics["test_loss"])
            elif self.args.dataset == "shakespeare":
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                total_loss = 0
                count = 0
                for batch_idx, (data, target) in enumerate(dataset):
                    #data = data.to(self.device)
                    #target = target.to(self.device)
                    output = model.forward(data)

                    # Discard the effective history, just like in training
                    eff_history = data.size(1)-1
                    final_output = output[:, eff_history:].contiguous().view(-1, self.args.n_chars)
                    final_target = target[:, eff_history:].contiguous().view(-1)

                    loss = criterion(final_output, final_target)

                    #need to verify this
                    _, predicted = torch.max(final_output, -1)
                    correct = predicted.eq(final_target).sum()
                    metrics["test_correct"] += correct.item()
                    metrics["test_total"] += final_target.size(0)
                    # Note that we don't add TAR loss here
                    total_loss += loss.data * final_output.size(0)
                    count += final_output.size(0)
                metrics["test_loss"] = float(total_loss.item()) / count * 1.0
            else:
                metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
                for batch_idx, (x, target) in enumerate(dataset):
                    x = x.to(self.device)
                    target = target.to(self.device)
                    pred = model.forward(x)
                    if self.args.model == "darts":
                        pred = pred[0]
                    loss = criterion(pred, target)

                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss.item() * target.size(0)
                    metrics["test_total"] += target.size(0)
        return metrics
