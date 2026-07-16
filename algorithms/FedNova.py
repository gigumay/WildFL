import sys
import json
import warnings
import shutil
import torch

from math import ceil
from typing import OrderedDict
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *


class FedNova:
    """
    Implementation of the FedNova algorithm (Wang et al. 2020). To compute the client weights at the server, I use the number
    of local annotations, and NOT the number of patches. I don't think there's a lot of difference between the two options,
    but using the number of samples allows me to compute an average weight across classes in a multi-case scenario (i.e., final
    weight = mean of weights for all classes). For normalizing the client updates, on the other hand, I have to use the number
    of patches, since they determine the number of local optimizer steps tau_i.
    """
    def __init__(self, rounds: int, mdl_type: str, mdl_version: str):
        """
        Initialize the FedNova class with model configuration and rounds.
        Arguments:
            rounds (int):       number of communication rounds.
            mdl_type (str):     type of model. can be yolo or polo.
            mdl_version (str):  model version (n/s/l/x).
        """
        self.rounds = rounds
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.server_mdl_state_file = None
        self.server_mdl = None
        self.client_states = {}
        self.temp_model_path = None
        self.tau_eff = None


    def _compute_global_distr(self, client_stats: dict) -> dict:
        """
        Computes the global class distribution by aggregating statistics from all clients.
        Arguments:
            client_stats (dict):    dictionary containing client data distribution.
        Returns:
            dict containing global class distribution.
        """

        # default value should be 0 anyway, but just to make sure
        global_distr = defaultdict(lambda: 0)

        for c in client_stats.keys():
            client_distr = client_stats[c]["data"]["train"]["anns"]

            for cat in client_distr.keys():
                global_distr[cat] += client_distr[cat]

        return global_distr


    def _compute_client_scaling(self, client_stats: dict, training_params: dict) -> dict:
        """
        Computes scaling factors for clients: the aggregation weight p_i (based on class distributions) and the
        number of local optimizer steps tau_i (based on the number of training patches), which is used to normalize
        the client updates. Also computes tau_eff = sum_i p_i * tau_i (the FedAvg-consistent choice in the paper),
        which rescales the aggregated normalized updates at the server.
        Note that tau_i is the plain local step count. Ultralytics trains with momentum SGD, for which the exact
        FedNova normalizer ||a_i||_1 differs from tau_i only by ~rho/(1-rho) (~9) steps - negligible here.
        Arguments:
            client_stats (dict):        dictionary containing client data statistics.
            training_params (dict):     client training parameters (needed for epochs per round and batch size).
        Returns:
            dict containing scaling factors for each client.
        """
        global_distr = self._compute_global_distr(client_stats=client_stats)

        if len(global_distr.keys()) != 1:
            warnings.warn("Client scaling strategy undefined for multi-class case. Naïvely averaging for now.")

        client_scaling = {c: {"server": 0, "normalization": 0} for c in client_stats.keys()}

        for c in client_stats.keys():
            client_distr = client_stats[c]["data"]["train"]["anns"]

            scaling = {cat: client_distr[cat] / global_distr[cat] for cat in client_distr.keys()}

            n_patches = client_stats[c]["data"]["train"]["patches"]["patches_kept"]

            client_scaling[c]["server"] = sum(scaling.values()) / len(scaling.keys())
            # local steps per round: steps per epoch (partial last batch included) times epochs per round
            client_scaling[c]["normalization"] = ceil(n_patches / training_params["batch"]) * training_params["epochs"]

        self.tau_eff = sum(client_scaling[c]["server"] * client_scaling[c]["normalization"] for c in client_scaling.keys())

        return client_scaling



    def _train_client(self, data_dir: str, client: str, out_put_dir: str, training_params: dict) -> dict:
        """
        Train a client model using the specified training parameters, and starting from the weights of the
        global model.
        Arguments:
            data_dir (str):             path to client data directory
            client (str):               client identifier
            out_put_dir (str):          directory to save training outputs
            training_params (dict):     training parameters
        Returns:
            state dict of trained client model
        """

        client_dataset = f"{data_dir}/{client}/{NAMES_DIRECTORIES_FILES['ultralytics_yaml']}"

        client_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
        client_mdl.train(data=client_dataset, state=self.server_mdl_state_file, name=client, project=out_put_dir, **training_params)

        return client_mdl.model.state_dict()


    def _aggregate_updates(self, client_updates: dict, client_scaling: dict) -> OrderedDict:
        """
        Aggregate client model updates according to the FedNova update rule (Eq. 12 in Wang et al. 2020):
            x_global_new = x_global + tau_eff * sum_i p_i * (Delta_i / tau_i)
        i.e., each client's cumulative update Delta_i (w.r.t. the broadcast global model) is normalized by its
        local step count tau_i and the aggregate is rescaled by tau_eff = sum_i p_i * tau_i. Clients that took
        many local steps are thereby down-weighted (rescaled by tau_eff/tau_i < 1) and vice versa; with equal
        tau_i for all clients, this reduces exactly to FedAvg.
        Arguments:
            client_updates (dict):      dictionary containing updated client models
            client_scaling (dict):      dictionary of client scaling factors (aggregation weight p_i and steps tau_i)
        Returns:
            state dict of updated global model.
        """
        global_state = torch.load(self.server_mdl_state_file)
        new_state = deepcopy(global_state)

        for k in new_state.keys():
            if "num_batches_tracked" in k:
                continue

            avg = torch.zeros_like(new_state[k])

            for c in client_updates.keys():
                assert client_updates[c][k].shape == new_state[k].shape, "Shape mismatch between client and server model."
                delta = client_updates[c][k] - global_state[k]
                avg += delta * (client_scaling[c]["server"] / client_scaling[c]["normalization"])

            new_state[k] += self.tau_eff * avg

        return new_state


    def _execute_round(self, round: int, clients: list, training_params: dict, data_dir: str,
                       output_dir: str, rm_client_data: bool = False) -> dict:
        """
        Execute a single round of communication and training.
        Arguments:
            round (int):                current round index
            clients (list):             list of client identifiers
            training_params (dict):     client training parameters
            data_dir (str):             path to the main data directory
            output_dir (str):           directory where training outputs can be stored
            rm_client_data (bool):      if True training outputs are removed after every round
        Returns:
            dict mapping each client to its locally-trained state dict.
        """

        round_dir = f"{output_dir}/round{round}"
        Path(round_dir).mkdir(exist_ok=True)

        client_updates = {c: None for c in clients}

        for c in clients:
            client_updates[c] = self._train_client(data_dir=data_dir, client=c, out_put_dir=round_dir, training_params=training_params)
            torch.save(client_updates[c], self.client_states[c])

        if rm_client_data:
            shutil.rmtree(round_dir)

        return client_updates


    def __call__(self, clients: list, data_dir: str, cfg_dir: str, class_id2name: dict, ann_format: str, training_params: dict, img_format: str,
                 eval_split: str, use_tiling: bool, output_dir: str, radii: dict = None, dor_eval: float = 0.3,
                 rm_client_data: bool = False, box_dims: dict = None, patch_dims: dict = None, patch_overlap: float = None, eval_global: bool = False) -> dict:
        """
        Runs federated training and evaluation for multiple rounds.
        The structure of data_dir must meet the following requirements:
        --data_dir
            --c1
                --eval_split
                ...
            --c2
                --eval_split
                ...
            ...
        with one subdirectory for each client whose datasets is part of the evaluation.
        Each eval_split subdirectory is further expected to contain an annotation json file and ground truth files for counting.
        Arguments:
            clients (list):                 list of client identifiers
            data_dir (str):                 path tothe main data directory. This directory is expected to be structured as specified above.
            cfg_yaml (str):                 path to the directory containing ultralytics configuration yaml files
            class_id2name (dict):           mapping of class IDs to class names
            ann_format (str):               annotation format
            training_params (dict):         dict containing training parameters
            img_format (str):               image file format
            eval_split (str):               split to use for model evaluation
            use_tiling (bool):              whether to perform tiled inference
            output_dir (str):               directory where outputs can be stored
            radii (dict):                   dict containign radii for each class
            dor_eval (float):               dor threshold to use for evaluation
            rm_client_data (bool):          if True client trainign data will be removed after every round
            box_dims (dict):                box dimensions used for training
            patch_dims (dict):              dict specifying the patch dimensions to use for tiled inference
            patch_overlap:                  overlap between patches to use for tiled inference
            eval_global (bool):             if True the aggregated global model is evaluated on all client datasets after every round

        Returns:
            dict containing performance metrics of the global model after every round
        """

        assert not training_params["val"], "Activating validation will lead to federation fo the best weights, not the last!"

        with open(f"{data_dir}/{NAMES_DIRECTORIES_FILES['FL_client_stats_json']}", "r") as f:
            client_stats = json.load(f)

        modify_cfg_yaml_nc(yaml_file=f"{cfg_dir}/{self.mdl_type}.yaml", nc=len(class_id2name.keys()))

        if "yolo" in self.mdl_type:
            task = "detect"
        elif "polo" in self.mdl_type:
            task = "locate"
        else:
            raise ValueError(f"Unknown model type '{self.mdl_type}'.")

        self.server_mdl_state_file = f"{output_dir}/server_mdl_state.pt"
        self.server_mdl = f"{output_dir}/server_mdl.pt"
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        # store a single reference state and start the federated process from there (cf. FedAvg)
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        torch.save(ref_state, self.server_mdl_state_file)

        self.client_states = {c: f"{output_dir}/{c}_state.pt" for c in clients}

        client_scaling = self._compute_client_scaling(client_stats=client_stats, training_params=training_params)

        global_results = {}
        local_results = {}

        datasets2test = deepcopy(clients)

        for i in range(self.rounds):

            client_updates = self._execute_round(round=i, clients=clients, training_params=training_params,
                                                 data_dir=data_dir, output_dir=output_dir, rm_client_data=rm_client_data)

            """
            I have to load a new model for every evaluation because evaluation will put it into eval mode, which changes the
            model structure (layers are fused) and thereby also the state dict, making parameter aggregation impossible.
            """

            """
            Evaluate the local client models FIRST, before aggregating them into the new global model. Each local model is
            tested only on its own dataset.
            """
            local_results[f"round{i}"] = {}
            for c in clients:
                local_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                local_mdl.model.load_state_dict(torch.load(self.client_states[c]))
                local_mdl.save(self.temp_model_path)

                local_results[f"round{i}"].update(eval_single_model(mdl=self.temp_model_path,
                                                                    task=task,
                                                                    data_dir=data_dir,
                                                                    img_format=img_format,
                                                                    datasets2test=[c],
                                                                    eval_split=eval_split,
                                                                    use_tiling=use_tiling,
                                                                    class_id2name=class_id2name,
                                                                    radii=radii,
                                                                    dor=dor_eval,
                                                                    ann_format=ann_format,
                                                                    box_dims=box_dims,
                                                                    patch_dims=patch_dims,
                                                                    patch_overlap=patch_overlap,
                                                                    output_dir=output_dir))

                Path(self.temp_model_path).unlink()

            # now aggregate the normalized local updates into the new global model
            new_state = self._aggregate_updates(client_updates=client_updates, client_scaling=client_scaling)
            Path(self.server_mdl_state_file).unlink()
            torch.save(new_state, self.server_mdl_state_file)

            # evaluate the global model on all client datasets
            if eval_global:
                global_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                global_state = torch.load(self.server_mdl_state_file)
                global_mdl.model.load_state_dict(global_state)
                global_mdl.save(self.server_mdl)

                global_results[f"round{i}"] = eval_single_model(mdl=self.server_mdl,
                                                                task=task,
                                                                data_dir=data_dir,
                                                                img_format=img_format,
                                                                datasets2test=datasets2test,
                                                                eval_split=eval_split,
                                                                use_tiling=use_tiling,
                                                                class_id2name=class_id2name,
                                                                radii=radii,
                                                                dor=dor_eval,
                                                                ann_format=ann_format,
                                                                box_dims=box_dims,
                                                                patch_dims=patch_dims,
                                                                patch_overlap=patch_overlap,
                                                                output_dir=output_dir)

                with open(f"{output_dir}/result_fednova.json", "w") as f:
                    json.dump(global_results, f, indent=1)

            with open(f"{output_dir}/result_fednova_local.json", "w") as f:
                json.dump(local_results, f, indent=1)

        return global_results
