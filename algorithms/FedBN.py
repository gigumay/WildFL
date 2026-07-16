import sys
import json
import warnings 
import shutil
import random
import torch 

from typing import OrderedDict
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *


class FedBN:
    """
    Implementation of the FedBN algorithm. To compute the client server weights, I use the number of local annotations, 
    and NOT the number of patches. I don't think there's a lot of difference between the two options, but using the 
    number of samples allows me to compute an average weight across classes in a multi-case scenario (i.e., final 
    weight = mean of weights for all classes).
    """
    def __init__(self, rounds: int, mdl_type: str, mdl_version: str):
        """
        Initialize the FedBN class with model configuration and rounds.
        Arguments:
            rounds (int):       number of communication rounds.
            mdl_type (str):     type of model. can be yolo or polo.
            mdl_version (str):  model version (n/s/l/x).
        """
        self.rounds = rounds
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.server_mdl_state_file = None
        self.client_states = {}
        self.client_local_states = {}
        self.temp_model_path = None


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
    

    def _compute_client_scaling(self, client_stats) -> dict:
        """
        Computes scaling factors for clients based on class distributions.
        Arguments:
            client_stats (dict):    dictionary containing client data statistics.
        Returns:
            dict containing scaling factors for each client.
        """
        global_distr = self._compute_global_distr(client_stats=client_stats)

        if len(global_distr.keys()) != 1:
            warnings.warn("Client scaling strategy undefined for multi-class case. Naïvely averaging for now.")

        client_scaling = defaultdict(lambda: 0.0)

        for c in client_stats.keys():
            client_distr = client_stats[c]["data"]["train"]["anns"]
            scaling = {cat: client_distr[cat] / global_distr[cat] for cat in client_distr.keys()}
            client_scaling[c] = sum(scaling.values()) / len(scaling.keys())

        return client_scaling



    def _train_client(self, data_dir: str, client: str, out_put_dir: str, training_params: dict) -> dict:
        """
        Train a client model using the specified training parameters, and starting from the weights of the
        client's personalized model (the reference state in the first round).
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
        client_mdl.train(data=client_dataset, state=self.client_states[client], name=client, project=out_put_dir, **training_params)

        return client_mdl.model.state_dict()


    def _aggregate_updates(self, client_updates: dict, client_scaling: dict) -> tuple[dict, OrderedDict]:
        """
        Aggregate client model updates using scaling factors and excluding BN parameters.
        Arguments:
            round (int):                index of current round
            client_updates (dict):     dictionary containing updated client models
            client_scaling (dict):      dictionary of client scaling factors
        Returns:
            Dict containing aggregated state dict for each client (clients keep individual BN params), and a state dict
            containing the non-BN params aggregated across clients.
        """
        agg_states = {}

        # aggregate non-BN params
        new_common_state = torch.load(self.server_mdl_state_file)

        for k in new_common_state.keys():
            if ".bn." in k:
               continue
           
            avg = torch.zeros_like(new_common_state[k])
            for c in client_updates.keys():
                assert client_updates[c][k].shape == new_common_state[k].shape, "Shape mismatch between client and server model."
                avg += (client_updates[c][k] * client_scaling[c])
            
            new_common_state[k] = avg

        # make individual state dict for each client with local BN params
        for c in client_updates.keys():
            client_sd = deepcopy(new_common_state)

            for k in client_sd.keys():
                if not ".bn." in k:
                    continue
                
                assert client_updates[c][k].shape == client_sd[k].shape, "Shape mismatch while loading BN params!"
                client_sd[k] = deepcopy(client_updates[c][k])
            
            agg_states[c] = client_sd

        return agg_states, new_common_state


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
            dict mapping each client to its locally-trained (pre-aggregation) state dict.
        """
        
        round_dir = f"{output_dir}/round{round}"
        Path(round_dir).mkdir(exist_ok=True)

        client_updates = {c: None for c in clients}

        for c in clients:
            client_updates[c] = self._train_client(data_dir=data_dir, client=c, out_put_dir=round_dir, training_params=training_params)
            torch.save(client_updates[c], self.client_local_states[c])

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
            data_dir (str):                 path tothe main data directory. This directory is expected to be structured as follows
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
            eval_global (bool):             if True the aggregated personalized models (shared params + local BN) are evaluated after every round

        Returns:
            dict containing performance metrics of the global model after every round
        """

        assert not training_params["val"], "Activating validation will lead to federation fo the best weights, not the last!"

        with open(f"{data_dir}/{NAMES_DIRECTORIES_FILES['FL_client_stats_json']}", "r") as f:
            client_stats = json.load(f)

        self.server_mdl_state_file = f"{output_dir}/server_mdl_state.pt"
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        modify_cfg_yaml_nc(yaml_file=f"{cfg_dir}/{self.mdl_type}.yaml", nc=len(class_id2name.keys()))

        # store a single reference state; seed the aggregation template and every client with it
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        torch.save(ref_state, self.server_mdl_state_file)

        self.client_states = {c: f"{output_dir}/{c}_state.pt" for c in clients}
        self.client_local_states = {c: f"{output_dir}/{c}_local_state.pt" for c in clients}
        for c in clients:
            torch.save(ref_state, self.client_states[c])

        client_scaling = self._compute_client_scaling(client_stats=client_stats)

        global_results = {}
        local_results = {}

        for i in range(self.rounds): 
            
            client_updates = self._execute_round(round=i, clients=clients, training_params=training_params,
                                                 data_dir=data_dir, output_dir=output_dir, rm_client_data=rm_client_data)

            """
            I have to load a new model for every evaluation because evaluation will put it into eval mode, which changes the
            model structure (layers are fused) and thereby also the state dict, making parameter aggregation impossible.
            """

            """
            Evaluate the local (pre-aggregation) client models FIRST, before averaging the shared parameters. Each local
            model is tested only on its own dataset.
            """
            local_results[f"round{i}"] = {}
            for c in clients:
                local_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                local_mdl.model.load_state_dict(torch.load(self.client_local_states[c]))
                local_mdl.save(self.temp_model_path)

                local_results[f"round{i}"].update(eval_single_model(mdl=self.temp_model_path,
                                                                    task="detect",
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

            """
            Aggregate the non-BN parameters across clients; each client keeps its own BN parameters (FedBN particularity).
            """
            agg_states, new_common_state = self._aggregate_updates(client_updates=client_updates, client_scaling=client_scaling)
            Path(self.server_mdl_state_file).unlink()
            torch.save(new_common_state, self.server_mdl_state_file)
            for c in clients:
                Path(self.client_states[c]).unlink()
                torch.save(agg_states[c], self.client_states[c])

            """
            Evaluate the aggregated personalized models (shared backbone + local BN parameters). Each is tested only on
            its own dataset.
            """
            if eval_global:
                global_results[f"round{i}"] = {}
                for c in clients:
                    agg_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                    agg_mdl.model.load_state_dict(torch.load(self.client_states[c]))
                    agg_mdl.save(self.temp_model_path)

                    global_results[f"round{i}"].update(eval_single_model(mdl=self.temp_model_path,
                                                                         task="detect",
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

                with open(f"{output_dir}/result_fedbn.json", "w") as f:
                    json.dump(global_results, f, indent=1)

            with open(f"{output_dir}/result_fedbn_local.json", "w") as f:
                json.dump(local_results, f, indent=1)

        return global_results
