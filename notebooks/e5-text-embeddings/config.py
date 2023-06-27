from optimum.graphcore import IPUConfig
import torch
import logging
import os

logger = logging.getLogger("e5")

configs = {
    1: {
        'inference_ipus_per_replica': 1,
        'inference_layers_per_ipu': [24],
        'inference_matmul_proportion': [0.1],
        'inference_replication_factor': 4
       },
    2: {
        'inference_ipus_per_replica': 2,
        'inference_layers_per_ipu': [10,14],
        'inference_matmul_proportion': [0.1,0.1],
        'inference_replication_factor': 2
       },
    4: {
        'inference_ipus_per_replica': 4,
        'inference_layers_per_ipu': [3,7,7,7],
        'inference_matmul_proportion': [0.1,0.15,0.15,0.15],
        'inference_replication_factor': 1
       }
}


def get_ipu_config(n_ipu, model_ipu, device_iterations, replication_factor=None, random_seed=None):
    ipu_config = IPUConfig.from_pretrained("Graphcore/bert-large-uncased").to_dict()
    
    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/")
    ipu_config['executable_cache_dir'] = executable_cache_dir
    
    if model_ipu in configs:
        for conf_name in configs[model_ipu]:
            ipu_config[conf_name] = configs[model_ipu][conf_name]
    else:
        logger.error("Only 1,2 or 4 IPUs (`model_ipu`) are supported for model pipelining. Replication will be used to ensure full POD utilisation")
        raise ValueError("Invalid number of IPUs for model")

    ipu_config['inference_replication_factor'] *= n_ipu // ipu_config['inference_replication_factor']
        
    if replication_factor:
        if replication_factor * model_ipu <= n_ipu:
            ipu_config['inference_replication_factor'] = replication_factor
        else:
            logger.error(f"Model cannot be replicated by custom replication factor: replication_factor ({replication_factor}) * model_ipu ({model_ipu}) not <= available_ipus(4)")
            raise ValueError("Model cannot be replicated over number of available IPUs")
                
    ipu_config['inference_device_iterations'] = device_iterations
    ipu_config['recompute_checkpoint_every_layer'] = False
    ipu_config['replicated_tensor_sharding'] = True
    ipu_config['enable_half_partials'] = True
    
    if random_seed:
        ipu_config['random_seed'] = random_seed 
        torch.manual_seed(random_seed)

    return IPUConfig.from_dict(ipu_config).eval()
    