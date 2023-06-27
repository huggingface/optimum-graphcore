from optimum.graphcore import IPUConfig
import torch
import logging

logger = logging.getLogger("e5")

def get_ipu_config(pod_type, n_ipu, device_iterations, replication_factor=None, random_seed=None):
    ipu_config = IPUConfig.from_pretrained("Graphcore/bert-large-uncased").to_dict()

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

    if n_ipu in configs:
        for conf_name in configs[n_ipu]:
            ipu_config[conf_name] = configs[n_ipu][conf_name]
    else:
        logger.error("Only 1,2 or 4 IPUs (`n_ipu`) are supported for model pipelining. Replication will be used to ensure full POD utilisation")
        raise ValueError("Invalid number of IPUs for model")

    if pod_type == 'pod4':
        if replication_factor:
            if replication_factor * n_ipu <= 4:
                ipu_config['inference_replication_factor'] = replication_factor
            else:
                logger.error(f"Model cannot be replicated by custom replication factor: replication_factor ({replication_factor}) * n_ipu ({n_ipu}) not <= available_ipus(4)")
                raise ValueError("Model cannot be replicated over number of available IPUs")
                
    elif pod_type == 'pod16':
        ipu_config['inference_replication_factor'] *= 4
        if replication_factor and replication_factor * n_ipu <= 16:
            ipu_config['inference_replication_factor'] = replication_factor
    
    ipu_config['inference_device_iterations'] = device_iterations
    ipu_config['recompute_checkpoint_every_layer'] = False
    ipu_config['replicated_tensor_sharding'] = True
    ipu_config['enable_half_partials'] = True

    if random_seed:
        ipu_config['random_seed'] = random_seed 
        torch.manual_seed(random_seed)

    return IPUConfig.from_dict(ipu_config).eval()
    