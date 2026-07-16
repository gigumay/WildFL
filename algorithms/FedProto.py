import sys
import json
import warnings 
import random
import shutil
import torch
import torchvision

from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *


class FedProto:
    """
    Implementation of the FedProto algorithm. To compute the client server weights, I use the number of local annotations, 
    and NOT the number of patches. I don't think there's a lot of difference between the two options, but using the 
    number of samples allows me to compute an average weight across classes in a multi-case scenario (i.e., final 
    weight = mean of weights for all classes).
    """
    def __init__(self, rounds: int, mdl_type: str, mdl_version: str):
        """
        Initialize the FedProto class with model configuration and rounds.
        Arguments:
            rounds (int):           number of communication rounds.
            mdl_type (str):         type of model. can be yolo or polo.
            mdl_version (str):      model version (n/s/l/x).
        """
        self.rounds = rounds
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.n_classes = None
        self.global_obj_protos = None
        self.client_dirs = {}
        self.client_states = {}
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


    def _train_client(self, round: int, data_dir: str, client: str, out_put_dir: str, training_params: dict) -> dict:
        """
        Train a client model using the specified training parameters. 
        Arguments:
            round (int):                current round
            data_dir (str):             path to client data directory
            client (str):               client identifier
            out_put_dir (str):          directory to save training outputs
            training_params (dict):     training parameters
        Returns:
            state dict of trained client model
        """

        align_prototypes = round > 0

        client_dataset = f"{data_dir}/{client}/{NAMES_DIRECTORIES_FILES['ultralytics_yaml']}"

        client_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
        client_mdl.train(data=client_dataset,
                         state=self.client_states[client],
                         name=client,
                         project=out_put_dir,
                         global_obj_protos=self.global_obj_protos,
                         align_prototypes=align_prototypes,
                         **training_params,
                         features_out_dir_train=self.client_dirs[client])

        return client_mdl.model.state_dict()

        

    def _generate_client_proto(self, client: str) -> torch.Tensor:
        """
        Generate local object prototypes for a client using the extracted features.
        Arguments:
            client (str):                  client identifier
        Returns:
            client prototypes
        """

        client_features = torch.load(f"{self.client_dirs[client]}/features.pt")

        local_proto = client_features.mean(dim=0, keepdim=True)

        return local_proto


    def _execute_round(self, round: int, clients: list, training_params: dict, data_dir: str, img_format: str, output_dir: str, rm_client_data: bool = False) -> dict:
        """
        Execute a single round of communication and training.
        Arguments:
            round (int):                current round index
            clients (list):             list of client identifiers
            training_params (dict):     client training parameters
            data_dir (str):             path to the main data directory
            img_format (str):           image format
            output_dir (str):           directory where training outputs can be stored
            rm_client_data (bool):      if True training outputs are removed after every round
        Returns:
            One dictionary containing the client prototypes.
        """

        out_dir = f"{output_dir}/round{round}"
        Path(out_dir).mkdir(exist_ok=True)

        client_protos = {c: None for c in clients}

        for c in clients:
            client_state = self._train_client(round=round,
                                              data_dir=data_dir,
                                              client=c,
                                              out_put_dir=out_dir,
                                              training_params=training_params)

            torch.save(client_state, self.client_states[c])

            client_protos[c] = self._generate_client_proto(client=c)

        if rm_client_data:
            shutil.rmtree(out_dir)

        return client_protos


    def _server_aggregation(self, client_prototypes: dict, client_scaling: dict) -> torch.Tensor:
        """
        Aggregate local prototypes from clients to form the global prototype.
        Arguments:
            client_prototypes (dict):   dictionary containing local prototypes from each client
            client_scaling (dict):      dictionary containing scaling factors for each client
        Returns:
            The global prototype.
        """

        client_protos_list = [client_prototypes[c] * client_scaling[c] for c in client_prototypes.keys()]
        client_protos_t = torch.cat(client_protos_list, dim=0)
        global_proto = client_protos_t.sum(dim=0, keepdim=True)

        return global_proto



    def _init_self(self, clients: list, training_params: dict, class_id2name: dict, output_dir: str) -> None:

        """
        Initialize internal variables and paths.
        Arguments:
            clients (list):                         list of client identifiers
            training_params (dict):                 dict containing training parameters
            class_id2name (dict):                   mapping of class IDs to class names
            output_dir (str):                       directory where outputs can be stored
        Returns:
            None
        """

        assert not training_params["val"], "Activating validation will lead to federation fo the best weights, not the last!"

        self.n_classes = len(class_id2name.keys())
      
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        self.client_dirs = {c: f"{output_dir}/{c}_data" for c in clients}
        for path in self.client_dirs.values():
             Path(path).mkdir(parents=True)

        self.client_states = {k: f"{v}/state.pt" for k,v in self.client_dirs.items()}
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        for c in self.client_states.keys():
            torch.save(ref_state, self.client_states[c])    

        # features are now a single set of tensors, so a single global object prototype file is used
        self.global_obj_protos = f"{output_dir}/global_obj_proto.pt"


    def __call__(self,
                 clients: list,
                 data_dir: str,
                 cfg_dir: str,
                 class_id2name: dict, 
                 ann_format: str,
                 training_params: dict,
                 inf_params: dict,
                 min_iou_eval: float,
                 img_format: str, 
                 eval_split: str, 
                 use_tiling: bool, 
                 output_dir: str, 
                 rm_client_data: bool = False, 
                 box_dims: dict = None,  
                 patch_dims: dict = None, 
                 patch_overlap: float = None,
                 measure_performance: bool = True) -> dict:
        """
        Runs federated training and evaluation for multiple rounds. The structure of data_dir must meet the following requirements:
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
            clients (list):                         list of client identifiers
            data_dir (str):                         path to the main data directory. This directory is expected to be structured as follows
            cfg_dir (str):                          path to the directory containing ultralytics configuration yaml files
            class_id2name (dict):                   mapping of class IDs to class names
            ann_format (str):                       annotation format
            training_params (dict):                 dict containing training parameters
            inf_params (dict):                      dict containing inference parameters for SAHI predictions
            min_iou_eval (float):                   minimum IoU threshold to consider a prediction as a true positive during evaluation.
            img_format (str):                       image file format
            eval_split (str):                       split to use for model evaluation
            use_tiling (bool):                      whether to perform tiled inference
            output_dir (str):                       directory where outputs can be stored
            rm_client_data (bool):                  if True client trainign data will be removed after every round
            box_dims (dict):                        box dimensions used for training
            patch_dims (dict):                      dict specifying the patch dimensions to use for tiled inference
            patch_overlap (foat):                   overlap between patches to use for tiled inference
            measure_performance (bool):             if True, model performance will be evaluated after every round
        Returns:
            dict containing performance metrics after every round
        """
        
        self._init_self(clients=clients,
                        training_params=training_params,
                        class_id2name=class_id2name,
                        output_dir=output_dir)
    
        with open(f"{data_dir}/{NAMES_DIRECTORIES_FILES['FL_client_stats_json']}", "r") as f:
            client_stats = json.load(f)
        
        client_scaling = self._compute_client_scaling(client_stats=client_stats)

        modify_cfg_yaml_nc(yaml_file=f"{cfg_dir}/{self.mdl_type}.yaml", nc=len(class_id2name.keys()))

        results = {}

        for i in range(self.rounds):
            client_obj_prototypes = self._execute_round(round=i, clients=clients, training_params=training_params, data_dir=data_dir,
                                                        img_format=img_format, output_dir=output_dir, rm_client_data=rm_client_data)

            print("Generating global prototypes...")
            global_obj_prototypes = self._server_aggregation(client_prototypes=client_obj_prototypes,
                                                             client_scaling=client_scaling)

            torch.save(global_obj_prototypes, self.global_obj_protos)

            results[f"round{i}"] = {}

            if measure_performance:
                for c in self.client_states.keys():
                    #I have to load a new model every round because evaluation will put it into eval mode, which changes the model structure 
                    #(layers are fused) and thereby also the state dict, making parameter aggregation in the subsequent round impossible.

                    client_model = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
                    client_state = torch.load(self.client_states[c])
                    client_model.model.load_state_dict(client_state)
                    client_model.save(self.temp_model_path)

                    res = eval_single_model(mdl=self.temp_model_path, 
                                            task="detect",  
                                            data_dir=data_dir, 
                                            img_format=img_format,
                                            datasets2test=[c], 
                                            eval_split=eval_split, 
                                            use_tiling=use_tiling,
                                            class_id2name=class_id2name,
                                            radii=None,
                                            dor=None,
                                            ann_format=ann_format,
                                            box_dims=box_dims, 
                                            patch_dims=patch_dims, 
                                            patch_overlap=patch_overlap, 
                                            output_dir=output_dir,
                                            inf_params=inf_params,
                                            min_iou=min_iou_eval)
                    results[f"round{i}"].update(res)


                    Path(self.temp_model_path).unlink()

            with open(f"{output_dir}/result_fedproto.json", "w") as f:
                json.dump(results, f, indent=1)
            
        return results
    
