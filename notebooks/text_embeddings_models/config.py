from optimum.graphcore import IPUConfig
import torch
import logging
import os

logger = logging.getLogger("e5")

default_config = {
    'inference_ipus_per_replica': 1,
    'inference_layers_per_ipu': [-1],
    'inference_matmul_proportion': [0.1],
    'inference_replication_factor': 4
   }
architectures = ['BertModel', 'MPNetModel', 'MPNetForMaskedLM', 'T5EncoderModel']

#TEMPORARY - Handle pretrained IPU config compatibility`
bert_large_uncased = {
  "device_iterations": 1,
  "embedding_serialization_factor": 2,
  "enable_half_first_order_momentum": True,
  "enable_half_partials": True,
  "executable_cache_dir": "./exe_cache",
  "gradient_accumulation_steps": 512,
  "inference_device_iterations": 4,
  "inference_replication_factor": 16,
  "ipus_per_replica": 4,
  "layers_per_ipu": [
    3,
    7,
    7,
    7
  ],
  "matmul_proportion": [
    0.1,
    0.15,
    0.15,
    0.15
  ],
  "optimizer_state_offchip": True,
  "optimum_version": "1.0.0",
  "output_mode": "final",
  "recompute_checkpoint_every_layer": True,
  "replicated_tensor_sharding": True,
  "replication_factor": 16,
  "seed": 42
}

#TEMPORARY - Handle pretrained IPU config compatibility`
bert_base_uncased = {
  "device_iterations": 1,
  "embedding_serialization_factor": 2,
  "enable_half_first_order_momentum": True,
  "enable_half_partials": True,
  "executable_cache_dir": "./exe_cache",
  "gradient_accumulation_steps": 512,
  "inference_device_iterations": 4,
  "inference_replication_factor": 4,
  "ipus_per_replica": 4,
  "layers_per_ipu": [
    0,
    4,
    4,
    4
  ],
  "matmul_proportion": 0.22,
  "optimizer_state_offchip": False,
  "optimum_version": "0.1.2",
  "output_mode": "final",
  "recompute_checkpoint_every_layer": True,
  "replicated_tensor_sharding": True,
  "replication_factor": 4,
  "seed": 42
}

def get_ipu_config(model_config, n_ipu, model_ipu, device_iterations, replication_factor=None, random_seed=None):
    base_architecture = model_config.architectures[0]
    
    if base_architecture not in architectures:
        logger.error(f"Model config passed does not contain a supported architecture: {architectures}")
        raise ValueError("Unsupported model architecture.")
    
    if model_ipu not in [1,4]:
        logger.error("Only 1 or 4 IPUs (`model_ipu`) are supported for model pipelining. Replication will be used to ensure full POD utilisation")
        raise ValueError("Invalid number of IPUs for model: {model_ipu}")

        
    # Set up number of layers for pipeline stages for E5 (Bert encoder) or MPNet (MPNet encoder) models
    if base_architecture == 'BertModel' or base_architecture == 'MPNetModel' or base_architecture == 'MPNetForMaskedLM':        
        if model_config.num_hidden_layers == 12:
            # ipu_config = IPUConfig.from_pretrained("Graphcore/bert-base-uncased").to_dict()
            ipu_config = bert_base_uncased
            if model_ipu == 1:
                ipu_config['inference_ipus_per_replica'] = 1
                ipu_config['inference_matmul_proportion'] = [0.1]
                ipu_config['inference_layers_per_ipu'] = [12]
                ipu_config['inference_replication_factor'] = 4
            elif model_ipu == 4:
                ipu_config['inference_replication_factor'] = 1
            
        elif model_config.num_hidden_layers == 24:
            # ipu_config = IPUConfig.from_pretrained("Graphcore/bert-large-uncased").to_dict()
            ipu_config = bert_large_uncased
            if model_ipu == 1:
                ipu_config['inference_ipus_per_replica'] = 1
                ipu_config['inference_matmul_proportion'] = [0.1]
                ipu_config['inference_layers_per_ipu'] = [24]
                ipu_config['inference_replication_factor'] = 4
            elif model_ipu == 4:
                ipu_config['inference_replication_factor'] = 1
            
        else:
            ipu_config = default_config
            if model_ipu == 1:
                ipu_config['inference_layers_per_ipu'] = [model_config.num_hidden_layers]
                ipu_config['inference_replication_factor'] = 4
            if model_ipu == 4:
                ipu_config['inference_ipus_per_replica'] = 4
                ipu_config['inference_layers_per_ipu'] = [-1,-1,-1,-1]
                ipu_config['inference_matmul_proportion'] = [0.1, 0.1, 0.1, 0.1]
                ipu_config['inference_replication_factor'] = -1

                
    # Set up number of layers for pipeline stages for Sentence-T5 (T5 encoder model)
    if base_architecture == 'T5EncoderModel':
        ipu_config = default_config
        if model_ipu == 1:
            ipu_config['inference_layers_per_ipu'] = [model_config.num_layers]
        if model_ipu == 4:
            ipu_config['inference_layers_per_ipu'] = [-1,-1,-1,-1]
            ipu_config['inference_ipus_per_replica'] = 4
            ipu_config['inference_matmul_proportion'] = [0.1,0.1,0.1,0.1]
            ipu_config['inference_replication_factor'] = 1

    
    # All other generic options
    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache/")
    ipu_config['executable_cache_dir'] = executable_cache_dir
    
    ipu_config['inference_replication_factor'] *= n_ipu // ipu_config['inference_replication_factor']
        
    if replication_factor:
        if replication_factor * model_ipu <= n_ipu:
            ipu_config['inference_replication_factor'] = replication_factor
        else:
            logger.error(f"Defined replication_factor ({replication_factor}) * model_ipu ({model_ipu}) not <= available_ipus(4)")
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

    print(ipu_config)
    return IPUConfig.from_dict(ipu_config).eval()
    