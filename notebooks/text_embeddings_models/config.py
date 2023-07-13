import logging
import os

import torch

from optimum.graphcore import IPUConfig


logger = logging.getLogger("e5")

default_config = {
    'inference_ipus_per_replica': 1,
    'inference_layers_per_ipu': [-1],
    'inference_matmul_proportion': [0.1],
    'inference_replication_factor': 4
   }
architectures = ['BertModel', 'MPNetModel', 'MPNetForMaskedLM', 'T5EncoderModel']

def get_ipu_config(model_config, n_ipu, ipus_per_replica, device_iterations, replication_factor=None, random_seed=None):
    base_architecture = model_config.architectures[0]

    if base_architecture not in architectures:
        logger.error(f"Model config passed does not contain a supported architecture: {architectures}")
        raise ValueError("Unsupported model architecture.")

    if ipus_per_replica not in [1,4]:
        logger.error("Only 1 or 4 IPUs (`ipus_per_replica`) are supported for model pipelining. Replication will be used to ensure full POD utilisation")
        raise ValueError("Invalid number of IPUs for model: {ipus_per_replica}")


    # Set up number of layers for pipeline stages for E5 (Bert encoder) or MPNet (MPNet encoder) models
    if base_architecture == 'BertModel' or base_architecture == 'MPNetModel' or base_architecture == 'MPNetForMaskedLM':
        if model_config.num_hidden_layers == 12:
            ipu_config = IPUConfig.from_pretrained("Graphcore/bert-base-uncased").to_dict()
            if ipus_per_replica == 1:
                ipu_config['inference_ipus_per_replica'] = 1
                ipu_config['inference_matmul_proportion'] = [0.2]
                ipu_config['inference_layers_per_ipu'] = [12]
                ipu_config['inference_replication_factor'] = 4
            elif ipus_per_replica == 4:
                ipu_config['inference_replication_factor'] = 1

        elif model_config.num_hidden_layers == 24:
            ipu_config = IPUConfig.from_pretrained("Graphcore/bert-large-uncased").to_dict()
            if ipus_per_replica == 1:
                ipu_config['inference_ipus_per_replica'] = 1
                ipu_config['inference_matmul_proportion'] = [0.1]
                ipu_config['inference_layers_per_ipu'] = [24]
                ipu_config['inference_replication_factor'] = 4
            elif ipus_per_replica == 4:
                ipu_config['inference_replication_factor'] = 1

        else:
            ipu_config = default_config
            if ipus_per_replica == 1:
                ipu_config['inference_layers_per_ipu'] = [model_config.num_hidden_layers]
                ipu_config['inference_replication_factor'] = 4
            if ipus_per_replica == 4:
                ipu_config['inference_ipus_per_replica'] = 4
                ipu_config['inference_layers_per_ipu'] = [-1,-1,-1,-1]
                ipu_config['inference_matmul_proportion'] = [0.1, 0.1, 0.1, 0.1]
                ipu_config['inference_replication_factor'] = -1


    # Set up number of layers for pipeline stages for Sentence-T5 (T5 encoder model)
    if base_architecture == 'T5EncoderModel':
        ipu_config = default_config
        if ipus_per_replica == 1:
            ipu_config['inference_layers_per_ipu'] = [model_config.num_layers]
        if ipus_per_replica == 4:
            ipu_config['inference_layers_per_ipu'] = [-1,-1,-1,-1]
            ipu_config['inference_ipus_per_replica'] = 4
            ipu_config['inference_matmul_proportion'] = [0.1,0.1,0.1,0.1]
            ipu_config['inference_replication_factor'] = 1


    # All other generic options
    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache/")
    ipu_config['executable_cache_dir'] = executable_cache_dir

    ipu_config['inference_replication_factor'] *= n_ipu // ipu_config['inference_replication_factor']

    if replication_factor:
        if replication_factor * ipus_per_replica <= n_ipu:
            ipu_config['inference_replication_factor'] = replication_factor
        else:
            logger.error(f"Defined replication_factor ({replication_factor}) * ipus_per_replica ({ipus_per_replica}) not <= available_ipus(4)")
            raise ValueError("Not enough IPUs for defined replication factor.")

    ipu_config['inference_device_iterations'] = device_iterations
    ipu_config['recompute_checkpoint_every_layer'] = False
    ipu_config['replicated_tensor_sharding'] = True
    ipu_config['enable_half_partials'] = True

    if 'embedding_serialization_factor' in ipu_config:
        if model_config.vocab_size % ipu_config['embedding_serialization_factor'] != 0:
            ipu_config['embedding_serialization_factor'] = 1

    if random_seed:
        ipu_config['seed'] = random_seed
        torch.manual_seed(random_seed)

    return IPUConfig.from_dict(ipu_config).eval()
