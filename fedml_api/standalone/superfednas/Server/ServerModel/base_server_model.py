from abc import ABC, abstractmethod
import wandb
import os
import torch
import numpy as np
from wandb.sdk.lib import RunDisabled
import copy
import random
import heapq


class BaseServerModel(ABC):
    def __init__(
        self, init_params, sampling_method, num_cli_total, cli_subnet_track=None,
    ):
        self.model = self.init_model(init_params)
        self.sampling_method = sampling_method
        if cli_subnet_track is None:
            self.cli_subnet_track = dict()
            for idx in range(num_cli_total):
                self.cli_subnet_track[idx] = dict()
                self.cli_subnet_track[idx]["largest"] = 0
                self.cli_subnet_track[idx]["smallest"] = 0
        else:
            self.cli_subnet_track = cli_subnet_track
        self.client_sample_count = None
        self.cli_indices = []
        self.top_k = 1
        self.bottom_k = 1
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        self.max_sample_count_idx = -1
        self.second_max_sample_count_idx = -1
        self.cur_round = -1
        self.cur_arch = None
        self.subnet_sampling = dict()
        self.subnet_sampling["static"] = self.static_sample
        self.subnet_sampling["dynamic"] = self.dynamic_sample
        self.subnet_sampling["all_random"] = self.random_subnet_sample
        self.subnet_sampling["compound"] = self.compound_subnet_sample
        self.subnet_sampling["sandwich_all_random"] = self.sandwich_all_subnet_sample
        self.subnet_sampling["sandwich_compound"] = self.sandwich_compound_subnet_sample
        self.subnet_sampling["TS_all_random"] = self.tracking_sandwich_all_subnet_sample
        self.subnet_sampling[
            "TS_compound"
        ] = self.tracking_sandwich_compound_subnet_sample
        self.subnet_sampling["max_sample_count"] = self.max_client_dataset_all_subnet
        self.subnet_sampling["multi_sandwich"] = self.multi_sandwich_sample
        self.subnet_sampling["TS_KD"] = self.tracking_sandwich_kd
        self.subnet_sampling["PS"] = self.ps

    def set_top_bottom_k(self, top_k, bottom_k):
        self.top_k = top_k
        self.bottom_k = bottom_k

    def load_client_sample_counts(self, sample_counts):
        self.client_sample_count = sample_counts

    def update_sample(self):
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        temp = set()
        heap = []
        available_cli = set()
        for cli in self.cli_indices:
            available_cli.add(cli)
        for i in available_cli:
            heapq.heappush(heap, self.cli_subnet_track[i]["largest"])
        for i in range(self.top_k):
            temp.add(heap[i])
        for cli in available_cli:
            if len(self.largest_subnet_min_idx) >= self.top_k:
                break
            if self.cli_subnet_track[cli]["largest"] in temp:
                self.largest_subnet_min_idx.add(cli)
        available_cli = available_cli - self.largest_subnet_min_idx
        # of the remaining clients temporally load balance min subnet
        heap = []
        for i in available_cli:
            heapq.heappush(heap, self.cli_subnet_track[i]["smallest"])
        temp = set()
        for i in range(self.bottom_k):
            temp.add(heap[i])
        for cli in available_cli:
            if len(self.smallest_subnet_min_idx) >= self.bottom_k:
                break
            if self.cli_subnet_track[cli]["smallest"] in temp:
                self.smallest_subnet_min_idx.add(cli)

        max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if (
                    max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[max_sample_count_idx]
            ):
                max_sample_count_idx = cur_idx
        # Find min smallest
        second_max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if cur_idx != max_sample_count_idx:
                if (
                    second_max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[second_max_sample_count_idx]
                ):
                    second_max_sample_count_idx = cur_idx
        self.max_sample_count_idx = max_sample_count_idx
        self.second_max_sample_count_idx = second_max_sample_count_idx

    def set_cli_indices(self, client_indices):
        self.cli_indices = client_indices

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model_copy(self):
        return copy.deepcopy(self.model)

    # superimpose a (numpy) vectorized version of it's MAX network onto supernetwork
    def superimpose_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    # sums supernetwork with a (numpy) vectorized version of it's MAX network in place
    def sum_supernet_w_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(p.data + vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def save(self, name, supernet_config, supernet_class):
        if isinstance(wandb.run, RunDisabled):
            return
        save_data = dict()
        save_data["params"] = self.get_model_params()
        save_data["supernet_config"] = supernet_config
        save_data["supernet_class"] = supernet_class
        save_data["torch_rng_state"] = torch.get_rng_state()
        save_data["numpy_rng_state"] = np.random.get_state()
        save_data["cli_subnet_track"] = self.cli_subnet_track
        filename = os.path.join(wandb.run.dir, name)
        torch.save(
            save_data, filename,
        )
        wandb.save(filename)

    #TODO: Abstract this function away and allow different supernetworks to specify expected input size
    def wandb_pass(self):
        self.model.set_max_net()
        self.model.eval()
        self.model(torch.zeros((1, 3, 32, 32)))

    def feddyn_global_model_update(self, alpha):
        for server_param, state_param in zip(
            self.model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / alpha) * state_param

    def sample_subnet(self, round_num, client_idx, idx, args):
        return self.subnet_sampling[self.sampling_method](
            round_num, client_idx, idx, args
        )

    def static_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str(client_idx % len(subnets))
        return self.get_subnet(**subnets[id])

    def dynamic_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str((client_idx + round_num) % len(subnets))
        return self.get_subnet(**subnets[id])

    def random_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def tracking_sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def tracking_sandwich_compound_subnet_sample(
        self, round_num, client_idx, idx, args
    ):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_compound_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def max_client_dataset_all_subnet(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.second_max_sample_count_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx == self.max_sample_count_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)
    def multi_sandwich_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = self.get_subnet(**self.min_subnet_arch())
        model_dict["mid"] = []
        for _ in range(args["K"]):
            model_dict["mid"].append(self.get_subnet(**self.random_subnet_arch()))
        return model_dict

    def tracking_sandwich_kd(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            model_dict = dict()
            model_dict["max"] = None
            model_dict["min"] = None
            model_dict["mid"] = [self.get_subnet(**self.max_subnet_arch())]
            self.cli_subnet_track[client_idx]["largest"] += 1
            return model_dict
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = None
        model_dict["mid"] = [arch_config]
        return model_dict

    def ps(self, round_num, client_idx, idx, args):
        if self.cur_round < round_num:
            self.cur_round = round_num
            #np.random.seed(round_num)
            if self.cur_round < args["ps_depth_only"]:
                self.cur_arch = self.random_depth_subnet_arch()
            else:
                self.cur_arch = self.random_subnet_arch()
        return self.get_subnet(**self.cur_arch)

    def crossover_sample(self, par1, par2):
        new_sample = copy.deepcopy(par1)
        for key in new_sample.keys():
            if not isinstance(new_sample[key], list):
                continue
            for i in range(len(new_sample[key])):
                new_sample[key][i] = random.choice([par1[key][i], par2[key][i]])
        return new_sample

    # TODO:Need to reimpliment updated sandwich sampling with dynamic width
    def dynamic_width_sandwich_sample(self, round_num, client_idx, idx, args):
        pass

    @abstractmethod
    def init_model(self, init_params):
        pass

    @abstractmethod
    def is_max_net(self, arch):
        pass

    @abstractmethod
    def is_min_net(self, arch):
        pass

    @abstractmethod
    def get_subnet(self, **kwargs):
        pass

    @abstractmethod
    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        pass

    @abstractmethod
    def active_subnet_index(self):
        pass

    @abstractmethod
    def max_subnet_arch(self):
        pass

    @abstractmethod
    def min_subnet_arch(self):
        pass

    @abstractmethod
    def random_subnet_arch(self):
        pass

    @abstractmethod
    def random_depth_subnet_arch(self):
        pass

    @abstractmethod
    def random_compound_subnet_arch(self):
        pass

    @abstractmethod
    def mutate_sample(self, sample_arch, mut_prob):
        pass
