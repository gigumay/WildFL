import sys
import json
import warnings 
import shutil
import gc
import torch 
import numpy as np
import torch.multiprocessing as mp

from typing import OrderedDict
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers import *



class MOON:
    """
    Implementation of the MOON algorithm. To compute the client weights, I use the number of local annotations, 
    and NOT the number of patches. I don't think there's a lot of difference between the two options, but using the 
    number of samples allows me to compute an average weight across classes in a multi-case scenario (i.e., final 
    weight = mean of weights for all classes).
    """
    def __init__(self, rounds: int, mdl_type: str, mdl_version: str):
        """
        Initialize the MOON class with model configuration and rounds.
        Arguments:
            rounds (int):       number of communication rounds.
            mdl_type (str):     type of model.
            mdl_version (str):  model version (n/s/l/x).
        """
        self.rounds = rounds
        self.mdl_type = mdl_type
        self.mdl_version = mdl_version
        self.server_mdl_state_file = None
        self.server_mdl = None
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


    
    def _prep_single_dir(self, dir: str, model: YOLO, embds_suffix: str, img_format: str) -> None:
        """
        Prepares a single directory for training by removing old embeddings and generating new ones.
        Arguments:
            dir (str):             path to the directory to be prepared.
            model (YOLO):          YOLO model instance for generating embeddings.
            embds_suffix (str):    suffix of the embedding files.
            img_format (str):      image file format.
        Returns:
            None
        """
        patch_files = list(Path(dir).glob(f"*.{img_format}"))
        assert len([pf for pf in patch_files if not Path(pf).with_suffix(".txt").exists()]) == 0, \
            "Found image file with no corresponding label file."
        
        embds = list(Path(dir).glob("*.npy"))
        embds2rm = [p for p in embds if embds_suffix in str(p)]


        for ef in embds2rm:
            ef.unlink()

        for pf in patch_files:
            e = model.predict(pf, embed=[16, 19, 22], verbose=False)
            #e = model.predict(pf, embed=[len], verbose=False)
            assert len(e) == 1, "Unexpected output dimensions!"

            emb_path = f"{dir}/{Path(pf).stem}{embds_suffix}.npy"
            np.save(emb_path, e[0].detach().cpu().numpy())


    
    def _prep_client(self, data_dir: str, client: str, use_global_mdl: bool, embds_suffix: str, img_format: str, model: OrderedDict = None) -> None:
        """
        Prepares the client data directory for training in the current or next round. At the start of a round this entails
        deleting previous global embeddings and generating new ones with thew updated version of the global model. At the end 
        of a round it means removing previous "previous" embeddings and generating new ones with the local model. 
        All of this is done for the training and validation directories of the client. 
        Arguments:
            data_dir (str):             path to the main data directory.
            client (str):               client identifier.
            use_global_mdl (bool):      if True, the global model is used to generate the embeddings. Must be called at
                                        the beginning of rounds.
            embds_suffix (str):         suffix of the embedding files that will be generated.
            img_format (str):           image file format.
            model (OrderedDict):        state dict of the model needed for generating embeddings. If None, the global model is used, 
                                        as should be the case at the start of a round.
        Returns:
            None
        """
        client_dir = f"{data_dir}/{client}"
        
        if use_global_mdl:
            mdl_state = torch.load(self.server_mdl_state_file)
        else:
            mdl_state = model
        
        mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
        mdl.model.load_state_dict(mdl_state)
            

        self._prep_single_dir(dir=f"{client_dir}/train", model=mdl, embds_suffix=embds_suffix, img_format=img_format)
        self._prep_single_dir(dir=f"{client_dir}/val", model=mdl, embds_suffix=embds_suffix, img_format=img_format)

        del mdl
        gc.collect()
        torch.cuda.empty_cache()
        for p in mp.active_children():
            p.terminate()

    def _train_client(self, data_dir: str, client: str, out_put_dir: str, training_params: dict) -> dict:
        """
        Train a client model using the specified training parameters, and starting from the weights of the 
        global model. Note that in the first round, the global embedding will be equal to the embeddings of the 
        model from the previous round. The contrastive loss will thus be constant (0.693).
        Arguments:
            data_dir (str):             path to client data directory
            client (str):               client identifier
            out_put_dir (str):          directory to save training outputs
            training_params (dict):     training parameters
        Returns:
            state dict of trained client model
        """
        state = self.server_mdl_state_file

        client_mdl = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml")
        client_dataset = f"{data_dir}/{client}/{NAMES_DIRECTORIES_FILES['ultralytics_yaml']}"
        client_mdl.train(data=client_dataset, state=state, name=client, project=out_put_dir, **training_params)

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
                       img_format: str, output_dir: str, rm_client_data: bool = False) -> dict:
        """
        Execute a single round of communication and training. This includes prepping the client dirs before and after training.
        Arguments:
            round (int):                current round index
            clients (list):             list of client identifiers
            training_params (dict):     client training parameters
            data_dir (str):             path to the main data directory
            img_format (str):           image file format
            output_dir (str):           directory where training outputs can be stored
            rm_client_data (bool):      if True training outputs are removed after every round
        Returns:
            dict mapping each client to its locally-trained state dict.
        """
        
        round_dir = f"{output_dir}/round{round}"
        Path(round_dir).mkdir()

        client_updates = {c: None for c in clients}

        for c in clients:
            caches = list(Path(f"{data_dir}/{c}").glob("*.cache"))
            for cache in caches:
                if cache.is_dir():
                    shutil.rmtree(cache)
                elif cache.is_file():
                    cache.unlink()
                else:
                    raise ValueError("Can't remove cache files!")

            self._prep_client(data_dir=data_dir, client=c, use_global_mdl=True, embds_suffix="_glob", img_format=img_format)
            if round == 0:
                self._prep_client(data_dir=data_dir, client=c, use_global_mdl=True, embds_suffix="_prev", img_format=img_format)
            client_updates[c] = self._train_client(data_dir=data_dir, client=c, out_put_dir=round_dir, training_params=training_params)
            torch.save(client_updates[c], self.client_states[c])
            self._prep_client(data_dir=data_dir, client=c, use_global_mdl=False, embds_suffix="_prev", img_format=img_format, model=client_updates[c])

        if rm_client_data:
            shutil.rmtree(round_dir)

        return client_updates


    def __call__(self, clients: list, data_dir: str, cfg_dir: str, class_id2name: dict, ann_format: str, training_params: dict, img_format: str,
                 eval_split: str, use_tiling: bool, output_dir: str, radii: dict = None, dor_eval: float = 0.3, rm_client_data: bool = False,
                 box_dims: dict = None, patch_dims: dict = None, patch_overlap: float = None, eval_global: bool = False) -> dict:
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

        self.server_mdl_state_file = f"{output_dir}/server_mdl_state.pt"
        self.server_mdl = f"{output_dir}/server_mdl.pt"
        self.temp_model_path = f"{output_dir}/temp_mdl.pt"

        # store a single reference state and start the federated process from there (cf. FedProto)
        ref_state = YOLO(f"{self.mdl_type}{self.mdl_version}.yaml").model.state_dict()
        torch.save(ref_state, self.server_mdl_state_file)

        self.client_states = {c: f"{output_dir}/{c}_state.pt" for c in clients}

        client_scaling = self._compute_client_scaling(client_stats=client_stats)

        global_results = {}
        local_results = {}

        datasets2test = deepcopy(clients)

        for i in range(self.rounds):
            client_updates = self._execute_round(round=i, clients=clients, training_params=training_params,
                                                 data_dir=data_dir, img_format=img_format, output_dir=output_dir, rm_client_data=rm_client_data)

            """
            I have to load a new model for every evaluation because evaluation will put it into eval mode, which changes the
            model structure (layers are fused) and thereby also the state dict, making parameter aggregation impossible.
            """

            """
            Evaluate the local client models FIRST, before averaging them into the new global model. Each local model is
            tested only on its own dataset.
            """
            local_results[f"round{i}"] = {}
            for c in clients:
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

            # evaluate the global model on all client datasets + all_data
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

                with open(f"{output_dir}/result_MOON.json", "w") as f:
                    json.dump(global_results, f, indent=1)

            with open(f"{output_dir}/result_MOON_local.json", "w") as f:
                json.dump(local_results, f, indent=1)

        return global_results
