import sys
import json
import shutil
import random
import torch 

from itertools import permutations
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *


class FedDC:
    """
    Implementation of the FedDC algorithm.
    """
    def __init__(self, agg_events: int, dc_period: int, agg_period: int, mdl_type: str, mdl_version: str):
        """
        Initialize the FedDC class with model configuration and aggregation parameters.
        Arguments:
            agg_events (int):   number of aggregation events to perform.
            dc_period (int):    frequency of daisy chaining.
            agg_period (int):   frequency of aggregation.
            mdl_type (str):     type of model. can be yolo or polo.
            mdl_version (str):  model version (n/s/l/x).
        """
        assert dc_period < agg_period, "Daisy chaining period must be shorter than aggregation period!"
        assert agg_period % dc_period == 0, "Aggregation period must be a multiple of daisy chaining period!"

        self.agg_events = agg_events
        self.dc_period = dc_period
        self.agg_period = agg_period
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.server_mdl_state_file = None
        self.server_mdl = None
        self.model_states = {}
        self.temp_model_path = None


    def _train_client(self, slot: int, pull_from_server: bool,  data_dir: str, client: str, out_put_dir: str, training_params: dict) -> None:
        """
        Train a client model using the specified training parameters, and starting from the weights of the 
        global model or from the training on another client. Writes the updated state dict to file.
        Arguments:
            slot (int):                 slot from which to pull the model for training. Updated 
                                        state dict will be written to the same slot.
            pull_from_server (bool):    if True trainign will start from the global model weights and 
                                        not from the weights returned by another client.
            data_dir (str):             path to client data directory
            client (str):               client identifier
            out_put_dir (str):          directory to save training outputs
            training_params (dict):     training parameters
        """
        assert "epochs" not in training_params.keys(), "Epochs should not be passed with the algorithm call!"

        sd_path = self.server_mdl_state_file if pull_from_server else self.model_states[slot]

        client_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
        client_dataset = f"{data_dir}/{client}/{NAMES_DIRECTORIES_FILES['ultralytics_yaml']}"
        client_mdl.train(data=client_dataset, state=sd_path, epochs=self.dc_period, name=client, project=out_put_dir, **training_params)

        Path(self.model_states[slot]).unlink(missing_ok=True)
        torch.save(client_mdl.model.state_dict(), self.model_states[slot])


    def _aggregate_updates(self) ->  None:
        sds = {k: torch.load(v) for k,v in self.model_states.items()}

        # aggregate params
        server_state = torch.load(self.server_mdl_state_file)

        for k in server_state.keys():
            if "num_batches_tracked" in k:
               continue
           
            avg = torch.zeros_like(server_state[k])
            for c in sds.keys():
                assert sds[c][k].shape == server_state[k].shape, "Shape mismatch between client and server model."
                avg += (sds[c][k] / len(list(sds.keys())))
            
            server_state[k] = avg

        Path(self.server_mdl_state_file).unlink(missing_ok=True)
        torch.save(server_state, self.server_mdl_state_file)


    def _run_agg_period(self, agg_event: int, clients: list, permutations: list, training_params: dict, data_dir: str, output_dir: str, 
                        rm_client_data: bool = False) -> list:
        """
        Run one aggregation period. Within one aggregation period, clients train their assigned models for epochs = daisy chaining period, 
        and then perform daisy chaining. This is repeated until the sum of daisy chaining periods is equal to the aggregation period. 
        Arguments:
            agg_event (int):            index of current aggregation event
            clients (list):             list of client identifiers
            permutations (list):        list of client permutations for daisy chaining.
            training_params (dict):     local training parameters
            data_dir (str):             path to the main data directory. 
            output_dir (str):           directory where outputs can be stored
            rm_client_data (bool):      if True client trainign data will be removed after every round
        Returns:
            list of client permutations that were chosen for daisy chaining.
        """
        
        agg_dir = f"{output_dir}/agg{agg_event}"
        Path(agg_dir).mkdir()

        perms = []

        n_dc = int(self.agg_period / self.dc_period)

        for i in range(n_dc):
            curr_perm = random.choice(permutations)
            perms.append(curr_perm)

            for c in clients:
                self._train_client(slot=curr_perm.index(c), pull_from_server=(i == 0), data_dir=data_dir, client=c, out_put_dir=agg_dir,
                                   training_params=training_params)

        if rm_client_data:
            shutil.rmtree(agg_dir)

        return perms


    def __call__(self, clients: list, data_dir: str, cfg_dir: str, class_id2name: dict, ann_format: str, training_params: dict, img_format: str,
                 eval_split: str, use_tiling: bool, output_dir: str, radii: dict = None, dor_eval: float = 0.3, rm_client_data: bool = False,
                 box_dims: dict = None, patch_dims: dict = None, patch_overlap: float = None, eval_global: bool = False) -> dict:
        """
        Runs federated training and evaluation for multiple rounds. Here one round = one aggregation event
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
            data_dir (str):                 path to the main data directory.
            cfg_dir (str):                  path to the directory containing ultralytics configuration yaml files 
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

        client_permutations = list(permutations(clients))
        perm_history = []

        modify_cfg_yaml_nc(yaml_file=f"{cfg_dir}/{self.mdl_type}.yaml", nc=len(class_id2name.keys()))

        self.server_mdl_state_file = f"{output_dir}/server_mdl_state.pt"
        self.server_mdl = f"{output_dir}/server_mdl.pt"
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        # store a single reference state and start the federated process from there (cf. FedAvg/FedProto)
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        torch.save(ref_state, self.server_mdl_state_file)

        for i in range(len(clients)):
            self.model_states[i] = f"{output_dir}/mdl{i}_state.pt"

        results = {}
        local_results = {}

        datasets2test = deepcopy(clients)

        for i in range(self.agg_events):
            perms = self._run_agg_period(agg_event=i, clients=clients, permutations=client_permutations, training_params=training_params, data_dir=data_dir, 
                                         output_dir=output_dir, rm_client_data=rm_client_data)
            
            perm_history.extend(perms)

            """
            I have to load a new model for every evaluation because evaluation will put it into eval mode, which changes the model structure
            (layers are fused) and thereby also the state dict, making parameter aggregation impossible.
            """

            """
            Evaluate the local client models FIRST, before averaging them into the new global model. Each client's local model is the one
            in the slot where its daisy chain ended (i.e., the model that was last trained on its data), and is tested only on its own dataset.
            """
            local_results[f"round{i}"] = {}
            for c in clients:
                local_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                local_mdl.model.load_state_dict(torch.load(self.model_states[perms[-1].index(c)]))
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
            self._aggregate_updates()

            # evaluate the global model on all client datasets
            if eval_global:
                global_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                global_state = torch.load(self.server_mdl_state_file)
                global_mdl.model.load_state_dict(global_state)
                global_mdl.save(self.server_mdl)

                results[f"round{i}"] = eval_single_model(mdl=self.server_mdl,
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

                with open(f"{output_dir}/result_fedDC.json", "w") as f:
                    json.dump(results, f, indent=1)

            with open(f"{output_dir}/result_fedDC_local.json", "w") as f:
                json.dump(local_results, f, indent=1)

        with open(f"{output_dir}/perm_history.json", "w") as f:
            json.dump(perm_history, f, indent=1)

        return results
    
