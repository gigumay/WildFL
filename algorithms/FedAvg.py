import sys
import json
import warnings 
import shutil
import torch 

from typing import OrderedDict
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *


class FedAvg:
    """
    Implementation of the FedAvg algorithm. To compute the client weights, I use the number of local annotations, 
    and NOT the number of patches. I don't think there's a lot of difference between the two options, but using the 
    number of samples allows me to compute an average weight across classes in a multi-case scenario (i.e., final 
    weight = mean of weights for all classes).
    """
    def __init__(self, mdl_type: str, mdl_version: str, rounds: int = None):
        """
        Initialize the FedAvg class with model configuration and rounds.
        Arguments:
            mdl_type (str):     type of model. can be yolo or polo.
            mdl_version (str):  model version (n/s/l/x).
            rounds (int):       optional; number of communication rounds. Can be omitted if a client_schedule
                                is passed to __call__, in which case the rounds are derived from the schedule.
        """
        self.rounds = rounds
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.server_mdl_state_file = None
        self.server_mdl = None
        self.client_states = {}
        self.temp_model_path = None
        self.client_schedule = None


    def _compute_global_distr(self, client_stats: dict, clients: list) -> dict:
        """
        Computes the global class distribution by aggregating statistics from the specified clients.
        Arguments:
            client_stats (dict):    dictionary containing client data distribution.
            clients (list):         clients to include in the aggregation.
        Returns:
            dict containing global class distribution.
        """

        # default value should be 0 anyway, but just to make sure
        global_distr = defaultdict(lambda: 0)

        for c in clients:
            client_distr = client_stats[c]["data"]["train"]["anns"]

            for cat in client_distr.keys():
                global_distr[cat] += client_distr[cat]

        return global_distr


    def _compute_client_scaling(self, client_stats: dict, clients: list) -> dict:
        """
        Computes scaling factors for the specified clients based on class distributions. Factors are
        normalized over exactly these clients (they sum to 1), so client_stats may contain additional
        clients that do not participate.
        Arguments:
            client_stats (dict):    dictionary containing client data statistics.
            clients (list):         clients to compute scaling factors for.
        Returns:
            dict containing scaling factors for each client.
        """
        global_distr = self._compute_global_distr(client_stats=client_stats, clients=clients)

        if len(global_distr.keys()) != 1:
            warnings.warn("Client scaling strategy undefined for multi-class case. Naïvely averaging for now.")

        # plain dict: looking up a client outside the normalization set must fail loudly, not yield 0.0
        client_scaling = {}

        for c in clients:
            client_distr = client_stats[c]["data"]["train"]["anns"]

            scaling = {cat: client_distr[cat] / global_distr[cat] for cat in client_distr.keys()}

            client_scaling[c] = sum(scaling.values()) / len(scaling.keys())

        return client_scaling


    def _validate_schedule(self, client_schedule: dict, clients: list) -> int:
        """
        Validate a client join schedule and derive the total number of federation rounds from it.
        Arguments:
            client_schedule (dict):     maps each client to {"join_round": int, "rounds": int}
            clients (list):             list of client identifiers
        Returns:
            total number of rounds: max(join_round + rounds) over all clients.
        """
        if set(client_schedule.keys()) != set(clients):
            raise ValueError(f"client_schedule keys {sorted(client_schedule.keys())} must match the clients "
                             f"list {sorted(clients)} exactly.")

        for c, entry in client_schedule.items():
            if not isinstance(entry["join_round"], int) or entry["join_round"] < 0:
                raise ValueError(f"Client {c}: 'join_round' must be an int >= 0, got {entry['join_round']}.")
            if not isinstance(entry["rounds"], int) or entry["rounds"] < 1:
                raise ValueError(f"Client {c}: 'rounds' must be an int >= 1, got {entry['rounds']}.")

        if min(entry["join_round"] for entry in client_schedule.values()) != 0:
            raise ValueError("At least one client must have join_round=0, otherwise round 0 has no participants.")

        return max(entry["join_round"] + entry["rounds"] for entry in client_schedule.values())


    def _get_active_clients(self, round: int, clients: list) -> list:
        """
        Determine which clients participate in the given round. Without a schedule, all clients do.
        With a schedule, a client participates from its join_round until the end of the federation.
        Arguments:
            round (int):        current round index
            clients (list):     list of all client identifiers
        Returns:
            list of participating clients (order of `clients` preserved).
        """
        if self.client_schedule is None:
            return clients

        return [c for c in clients if self.client_schedule[c]["join_round"] <= round]



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
        Aggregate client model updates using scaling factors.
        Arguments:
            client_updates (dict):     dictionary containing updated client models
            client_scaling (dict):      dictionary of client scaling factors
        Returns:
            state dict of updated global model.
        """
        new_state = torch.load(self.server_mdl_state_file)

        for k in new_state.keys():
            if "num_batches_tracked" in k:
                continue
            
            avg = torch.zeros_like(new_state[k])

            for c in client_updates.keys():
                assert client_updates[c][k].shape == new_state[k].shape, "Shape mismatch between client and server model."
                avg += (client_updates[c][k] * client_scaling[c])
            
            new_state[k] = avg

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
                 rm_client_data: bool = False, box_dims: dict = None, patch_dims: dict = None, patch_overlap: float = None, eval_global: bool = False,
                 client_schedule: dict = None) -> dict:
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
            client_schedule (dict):         optional; maps each client to {"join_round": int, "rounds": int}. A client participates
                                            from its join_round until the end of the federation. The total number of rounds is derived
                                            as max(join_round + rounds) over all clients and overrides the constructor's rounds argument
                                            (which can then be omitted). If no schedule is passed, rounds must be set in the constructor.

        Returns:
            dict containing performance metrics of the global model after every round
        """

        assert not training_params["val"], "Activating validation will lead to federation fo the best weights, not the last!"

        with open(f"{data_dir}/{NAMES_DIRECTORIES_FILES['FL_client_stats_json']}", "r") as f:
            client_stats = json.load(f)

        missing = [c for c in clients if c not in client_stats]
        if missing:
            raise KeyError(f"Clients {missing} not found in client_stats.json")

        self.client_schedule = client_schedule
        if client_schedule is not None:
            derived_rounds = self._validate_schedule(client_schedule=client_schedule, clients=clients)
            if self.rounds is not None and derived_rounds != self.rounds:
                warnings.warn(f"client_schedule implies {derived_rounds} rounds; overriding rounds={self.rounds}.")
            self.rounds = derived_rounds
        elif self.rounds is None:
            raise ValueError("Either rounds (constructor) or client_schedule (__call__) must be provided.")

        modify_cfg_yaml_nc(yaml_file=f"{cfg_dir}/{self.mdl_type}.yaml", nc=len(class_id2name.keys()))

        self.server_mdl_state_file = f"{output_dir}/server_mdl_state.pt"
        self.server_mdl = f"{output_dir}/server_mdl.pt"
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        # store a single reference state and start the federated process from there (cf. FedProto)
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        torch.save(ref_state, self.server_mdl_state_file)

        self.client_states = {c: f"{output_dir}/{c}_state.pt" for c in clients}

        global_results = {}
        local_results = {}

        datasets2test = deepcopy(clients)

        for i in range(self.rounds):

            # scaling is re-normalized every round over exactly the clients participating in it, so a joining
            # client shifts the weights of all others and the weights always sum to 1
            active_clients = self._get_active_clients(round=i, clients=clients)
            client_scaling = self._compute_client_scaling(client_stats=client_stats, clients=active_clients)

            client_updates = self._execute_round(round=i, clients=active_clients, training_params=training_params,
                                                 data_dir=data_dir, output_dir=output_dir, rm_client_data=rm_client_data)

            """
            I have to load a new model for every evaluation because evaluation will put it into eval mode, which changes the
            model structure (layers are fused) and thereby also the state dict, making parameter aggregation impossible.
            """

            """
            Evaluate the local client models FIRST, before averaging them into the new global model. Each local model is
            tested only on its own dataset.
            """
            local_results[f"round{i}"] = {}
            for c in active_clients:
                local_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                local_mdl.model.load_state_dict(torch.load(self.client_states[c]))
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

            # now aggregate the local models into the new global model
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
                                                                task="detect",
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

                with open(f"{output_dir}/result_fedavg.json", "w") as f:
                    json.dump(global_results, f, indent=1)

            with open(f"{output_dir}/result_fedavg_local.json", "w") as f:
                json.dump(local_results, f, indent=1)

        return global_results
