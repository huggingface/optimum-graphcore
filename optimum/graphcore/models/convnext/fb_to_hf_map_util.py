import re
import warnings

fb_pattern_str = "^(((downsample_layers\.([0-3]+)\.([0-1]+)\.(weight|bias)$)|(stages\.([0-9]+)\.([0-9]+)\.(gamma$|((dwconv|norm|pwconv1|pwconv2)\.(weight|bias)$)))|((norm|head).(weight|bias)$))|(.*$))"
re_fb_pattern = re.compile(fb_pattern_str)

def fb_to_hf_name(fb_name: str):
    # match regex
    matches = re_fb_pattern.search(fb_name)

    translated_name = ""
    
    if matches.group(2):
        model_component = matches.group(1)

        if matches.group(3):
            # downsample layers
            # of the form "downsample_layers.0.0.weight"
            stage = int(matches.group(4))
            block = int(matches.group(5))
            param_name = matches.group(6)
            if stage == 0:
                translated_name += "convnext.embeddings."
                if block == 0:
                    translated_name += f"patch_embeddings.{param_name}"
                elif block == 1: 
                    translated_name += f"layernorm.{param_name}"
            else:
                translated_name += f"convnext.encoder.stages.{stage}.downsampling_layer.{block}.{param_name}"
            return translated_name

        elif matches.group(7):
            # main encoder layers
            # of the form "stages.0.0.dwconv.weight"
            stage = int(matches.group(8))
            block = int(matches.group(9))
            translated_name += f"convnext.encoder.stages.{stage}.layers.{block}."

            if matches.group(11):
                op_name = matches.group(12)
                param_name = matches.group(13)

                if op_name == "norm":
                    op_name = "layernorm"

                translated_name += f"{op_name}.{param_name}"
            else:
                assert matches.group(10) == "gamma", f"expecting parameter name to be gamma: {fb_name}"
                translated_name += "layer_scale_parameter"

            return translated_name

        elif matches.group(14):
            # final layers
            # of the form "norm.weight" or "head.weight"
            op_name = matches.group(15)
            param_name = matches.group(16)

            if op_name == "norm":
                translated_name += f"convnext.layernorm.{param_name}"
            elif op_name == "head":
                translated_name += f"classifier.{param_name}"
            
            return translated_name

    else:
        warnings.warn(f"name does not match expected format: {fb_name} (re={fb_pattern_str})")
        return None


FB_TO_HF_MAP={"downsample_layers.0.0.weight":"convnext.embeddings.patch_embeddings.weight",
"downsample_layers.0.0.bias":"convnext.embeddings.patch_embeddings.bias",
"downsample_layers.0.1.weight":"convnext.embeddings.layernorm.weight",
"downsample_layers.0.1.bias":"convnext.embeddings.layernorm.bias",
"downsample_layers.1.0.weight":"convnext.encoder.stages.1.downsampling_layer.0.weight",
"downsample_layers.1.0.bias":"convnext.encoder.stages.1.downsampling_layer.0.bias",
"downsample_layers.1.1.weight":"convnext.encoder.stages.1.downsampling_layer.1.weight",
"downsample_layers.1.1.bias":"convnext.encoder.stages.1.downsampling_layer.1.bias",
"downsample_layers.2.0.weight":"convnext.encoder.stages.2.downsampling_layer.0.weight",
"downsample_layers.2.0.bias":"convnext.encoder.stages.2.downsampling_layer.0.bias",
"downsample_layers.2.1.weight":"convnext.encoder.stages.2.downsampling_layer.1.weight",
"downsample_layers.2.1.bias":"convnext.encoder.stages.2.downsampling_layer.1.bias",
"downsample_layers.3.0.weight":"convnext.encoder.stages.3.downsampling_layer.0.weight",
"downsample_layers.3.0.bias":"convnext.encoder.stages.3.downsampling_layer.0.bias",
"downsample_layers.3.1.weight":"convnext.encoder.stages.3.downsampling_layer.1.weight",
"downsample_layers.3.1.bias":"convnext.encoder.stages.3.downsampling_layer.1.bias",
"stages.0.0.gamma":"convnext.encoder.stages.0.layers.0.layer_scale_parameter",
"stages.0.0.dwconv.weight":"convnext.encoder.stages.0.layers.0.dwconv.weight",
"stages.0.0.dwconv.bias":"convnext.encoder.stages.0.layers.0.dwconv.bias",
"stages.0.0.norm.weight":"convnext.encoder.stages.0.layers.0.layernorm.weight",
"stages.0.0.norm.bias":"convnext.encoder.stages.0.layers.0.layernorm.bias",
"stages.0.0.pwconv1.weight":"convnext.encoder.stages.0.layers.0.pwconv1.weight",
"stages.0.0.pwconv1.bias":"convnext.encoder.stages.0.layers.0.pwconv1.bias",
"stages.0.0.pwconv2.weight":"convnext.encoder.stages.0.layers.0.pwconv2.weight",
"stages.0.0.pwconv2.bias":"convnext.encoder.stages.0.layers.0.pwconv2.bias",
"stages.0.1.gamma":"convnext.encoder.stages.0.layers.1.layer_scale_parameter",
"stages.0.1.dwconv.weight":"convnext.encoder.stages.0.layers.1.dwconv.weight",
"stages.0.1.dwconv.bias":"convnext.encoder.stages.0.layers.1.dwconv.bias",
"stages.0.1.norm.weight":"convnext.encoder.stages.0.layers.1.layernorm.weight",
"stages.0.1.norm.bias":"convnext.encoder.stages.0.layers.1.layernorm.bias",
"stages.0.1.pwconv1.weight":"convnext.encoder.stages.0.layers.1.pwconv1.weight",
"stages.0.1.pwconv1.bias":"convnext.encoder.stages.0.layers.1.pwconv1.bias",
"stages.0.1.pwconv2.weight":"convnext.encoder.stages.0.layers.1.pwconv2.weight",
"stages.0.1.pwconv2.bias":"convnext.encoder.stages.0.layers.1.pwconv2.bias",
"stages.0.2.gamma":"convnext.encoder.stages.0.layers.2.layer_scale_parameter",
"stages.0.2.dwconv.weight":"convnext.encoder.stages.0.layers.2.dwconv.weight",
"stages.0.2.dwconv.bias":"convnext.encoder.stages.0.layers.2.dwconv.bias",
"stages.0.2.norm.weight":"convnext.encoder.stages.0.layers.2.layernorm.weight",
"stages.0.2.norm.bias":"convnext.encoder.stages.0.layers.2.layernorm.bias",
"stages.0.2.pwconv1.weight":"convnext.encoder.stages.0.layers.2.pwconv1.weight",
"stages.0.2.pwconv1.bias":"convnext.encoder.stages.0.layers.2.pwconv1.bias",
"stages.0.2.pwconv2.weight":"convnext.encoder.stages.0.layers.2.pwconv2.weight",
"stages.0.2.pwconv2.bias":"convnext.encoder.stages.0.layers.2.pwconv2.bias",
"stages.1.0.gamma":"convnext.encoder.stages.1.layers.0.layer_scale_parameter",
"stages.1.0.dwconv.weight":"convnext.encoder.stages.1.layers.0.dwconv.weight",
"stages.1.0.dwconv.bias":"convnext.encoder.stages.1.layers.0.dwconv.bias",
"stages.1.0.norm.weight":"convnext.encoder.stages.1.layers.0.layernorm.weight",
"stages.1.0.norm.bias":"convnext.encoder.stages.1.layers.0.layernorm.bias",
"stages.1.0.pwconv1.weight":"convnext.encoder.stages.1.layers.0.pwconv1.weight",
"stages.1.0.pwconv1.bias":"convnext.encoder.stages.1.layers.0.pwconv1.bias",
"stages.1.0.pwconv2.weight":"convnext.encoder.stages.1.layers.0.pwconv2.weight",
"stages.1.0.pwconv2.bias":"convnext.encoder.stages.1.layers.0.pwconv2.bias",
"stages.1.1.gamma":"convnext.encoder.stages.1.layers.1.layer_scale_parameter",
"stages.1.1.dwconv.weight":"convnext.encoder.stages.1.layers.1.dwconv.weight",
"stages.1.1.dwconv.bias":"convnext.encoder.stages.1.layers.1.dwconv.bias",
"stages.1.1.norm.weight":"convnext.encoder.stages.1.layers.1.layernorm.weight",
"stages.1.1.norm.bias":"convnext.encoder.stages.1.layers.1.layernorm.bias",
"stages.1.1.pwconv1.weight":"convnext.encoder.stages.1.layers.1.pwconv1.weight",
"stages.1.1.pwconv1.bias":"convnext.encoder.stages.1.layers.1.pwconv1.bias",
"stages.1.1.pwconv2.weight":"convnext.encoder.stages.1.layers.1.pwconv2.weight",
"stages.1.1.pwconv2.bias":"convnext.encoder.stages.1.layers.1.pwconv2.bias",
"stages.1.2.gamma":"convnext.encoder.stages.1.layers.2.layer_scale_parameter",
"stages.1.2.dwconv.weight":"convnext.encoder.stages.1.layers.2.dwconv.weight",
"stages.1.2.dwconv.bias":"convnext.encoder.stages.1.layers.2.dwconv.bias",
"stages.1.2.norm.weight":"convnext.encoder.stages.1.layers.2.layernorm.weight",
"stages.1.2.norm.bias":"convnext.encoder.stages.1.layers.2.layernorm.bias",
"stages.1.2.pwconv1.weight":"convnext.encoder.stages.1.layers.2.pwconv1.weight",
"stages.1.2.pwconv1.bias":"convnext.encoder.stages.1.layers.2.pwconv1.bias",
"stages.1.2.pwconv2.weight":"convnext.encoder.stages.1.layers.2.pwconv2.weight",
"stages.1.2.pwconv2.bias":"convnext.encoder.stages.1.layers.2.pwconv2.bias",
"stages.2.0.gamma":"convnext.encoder.stages.2.layers.0.layer_scale_parameter",
"stages.2.0.dwconv.weight":"convnext.encoder.stages.2.layers.0.dwconv.weight",
"stages.2.0.dwconv.bias":"convnext.encoder.stages.2.layers.0.dwconv.bias",
"stages.2.0.norm.weight":"convnext.encoder.stages.2.layers.0.layernorm.weight",
"stages.2.0.norm.bias":"convnext.encoder.stages.2.layers.0.layernorm.bias",
"stages.2.0.pwconv1.weight":"convnext.encoder.stages.2.layers.0.pwconv1.weight",
"stages.2.0.pwconv1.bias":"convnext.encoder.stages.2.layers.0.pwconv1.bias",
"stages.2.0.pwconv2.weight":"convnext.encoder.stages.2.layers.0.pwconv2.weight",
"stages.2.0.pwconv2.bias":"convnext.encoder.stages.2.layers.0.pwconv2.bias",
"stages.2.1.gamma":"convnext.encoder.stages.2.layers.1.layer_scale_parameter",
"stages.2.1.dwconv.weight":"convnext.encoder.stages.2.layers.1.dwconv.weight",
"stages.2.1.dwconv.bias":"convnext.encoder.stages.2.layers.1.dwconv.bias",
"stages.2.1.norm.weight":"convnext.encoder.stages.2.layers.1.layernorm.weight",
"stages.2.1.norm.bias":"convnext.encoder.stages.2.layers.1.layernorm.bias",
"stages.2.1.pwconv1.weight":"convnext.encoder.stages.2.layers.1.pwconv1.weight",
"stages.2.1.pwconv1.bias":"convnext.encoder.stages.2.layers.1.pwconv1.bias",
"stages.2.1.pwconv2.weight":"convnext.encoder.stages.2.layers.1.pwconv2.weight",
"stages.2.1.pwconv2.bias":"convnext.encoder.stages.2.layers.1.pwconv2.bias",
"stages.2.2.gamma":"convnext.encoder.stages.2.layers.2.layer_scale_parameter",
"stages.2.2.dwconv.weight":"convnext.encoder.stages.2.layers.2.dwconv.weight",
"stages.2.2.dwconv.bias":"convnext.encoder.stages.2.layers.2.dwconv.bias",
"stages.2.2.norm.weight":"convnext.encoder.stages.2.layers.2.layernorm.weight",
"stages.2.2.norm.bias":"convnext.encoder.stages.2.layers.2.layernorm.bias",
"stages.2.2.pwconv1.weight":"convnext.encoder.stages.2.layers.2.pwconv1.weight",
"stages.2.2.pwconv1.bias":"convnext.encoder.stages.2.layers.2.pwconv1.bias",
"stages.2.2.pwconv2.weight":"convnext.encoder.stages.2.layers.2.pwconv2.weight",
"stages.2.2.pwconv2.bias":"convnext.encoder.stages.2.layers.2.pwconv2.bias",
"stages.2.3.gamma":"convnext.encoder.stages.2.layers.3.layer_scale_parameter",
"stages.2.3.dwconv.weight":"convnext.encoder.stages.2.layers.3.dwconv.weight",
"stages.2.3.dwconv.bias":"convnext.encoder.stages.2.layers.3.dwconv.bias",
"stages.2.3.norm.weight":"convnext.encoder.stages.2.layers.3.layernorm.weight",
"stages.2.3.norm.bias":"convnext.encoder.stages.2.layers.3.layernorm.bias",
"stages.2.3.pwconv1.weight":"convnext.encoder.stages.2.layers.3.pwconv1.weight",
"stages.2.3.pwconv1.bias":"convnext.encoder.stages.2.layers.3.pwconv1.bias",
"stages.2.3.pwconv2.weight":"convnext.encoder.stages.2.layers.3.pwconv2.weight",
"stages.2.3.pwconv2.bias":"convnext.encoder.stages.2.layers.3.pwconv2.bias",
"stages.2.4.gamma":"convnext.encoder.stages.2.layers.4.layer_scale_parameter",
"stages.2.4.dwconv.weight":"convnext.encoder.stages.2.layers.4.dwconv.weight",
"stages.2.4.dwconv.bias":"convnext.encoder.stages.2.layers.4.dwconv.bias",
"stages.2.4.norm.weight":"convnext.encoder.stages.2.layers.4.layernorm.weight",
"stages.2.4.norm.bias":"convnext.encoder.stages.2.layers.4.layernorm.bias",
"stages.2.4.pwconv1.weight":"convnext.encoder.stages.2.layers.4.pwconv1.weight",
"stages.2.4.pwconv1.bias":"convnext.encoder.stages.2.layers.4.pwconv1.bias",
"stages.2.4.pwconv2.weight":"convnext.encoder.stages.2.layers.4.pwconv2.weight",
"stages.2.4.pwconv2.bias":"convnext.encoder.stages.2.layers.4.pwconv2.bias",
"stages.2.5.gamma":"convnext.encoder.stages.2.layers.5.layer_scale_parameter",
"stages.2.5.dwconv.weight":"convnext.encoder.stages.2.layers.5.dwconv.weight",
"stages.2.5.dwconv.bias":"convnext.encoder.stages.2.layers.5.dwconv.bias",
"stages.2.5.norm.weight":"convnext.encoder.stages.2.layers.5.layernorm.weight",
"stages.2.5.norm.bias":"convnext.encoder.stages.2.layers.5.layernorm.bias",
"stages.2.5.pwconv1.weight":"convnext.encoder.stages.2.layers.5.pwconv1.weight",
"stages.2.5.pwconv1.bias":"convnext.encoder.stages.2.layers.5.pwconv1.bias",
"stages.2.5.pwconv2.weight":"convnext.encoder.stages.2.layers.5.pwconv2.weight",
"stages.2.5.pwconv2.bias":"convnext.encoder.stages.2.layers.5.pwconv2.bias",
"stages.2.6.gamma":"convnext.encoder.stages.2.layers.6.layer_scale_parameter",
"stages.2.6.dwconv.weight":"convnext.encoder.stages.2.layers.6.dwconv.weight",
"stages.2.6.dwconv.bias":"convnext.encoder.stages.2.layers.6.dwconv.bias",
"stages.2.6.norm.weight":"convnext.encoder.stages.2.layers.6.layernorm.weight",
"stages.2.6.norm.bias":"convnext.encoder.stages.2.layers.6.layernorm.bias",
"stages.2.6.pwconv1.weight":"convnext.encoder.stages.2.layers.6.pwconv1.weight",
"stages.2.6.pwconv1.bias":"convnext.encoder.stages.2.layers.6.pwconv1.bias",
"stages.2.6.pwconv2.weight":"convnext.encoder.stages.2.layers.6.pwconv2.weight",
"stages.2.6.pwconv2.bias":"convnext.encoder.stages.2.layers.6.pwconv2.bias",
"stages.2.7.gamma":"convnext.encoder.stages.2.layers.7.layer_scale_parameter",
"stages.2.7.dwconv.weight":"convnext.encoder.stages.2.layers.7.dwconv.weight",
"stages.2.7.dwconv.bias":"convnext.encoder.stages.2.layers.7.dwconv.bias",
"stages.2.7.norm.weight":"convnext.encoder.stages.2.layers.7.layernorm.weight",
"stages.2.7.norm.bias":"convnext.encoder.stages.2.layers.7.layernorm.bias",
"stages.2.7.pwconv1.weight":"convnext.encoder.stages.2.layers.7.pwconv1.weight",
"stages.2.7.pwconv1.bias":"convnext.encoder.stages.2.layers.7.pwconv1.bias",
"stages.2.7.pwconv2.weight":"convnext.encoder.stages.2.layers.7.pwconv2.weight",
"stages.2.7.pwconv2.bias":"convnext.encoder.stages.2.layers.7.pwconv2.bias",
"stages.2.8.gamma":"convnext.encoder.stages.2.layers.8.layer_scale_parameter",
"stages.2.8.dwconv.weight":"convnext.encoder.stages.2.layers.8.dwconv.weight",
"stages.2.8.dwconv.bias":"convnext.encoder.stages.2.layers.8.dwconv.bias",
"stages.2.8.norm.weight":"convnext.encoder.stages.2.layers.8.layernorm.weight",
"stages.2.8.norm.bias":"convnext.encoder.stages.2.layers.8.layernorm.bias",
"stages.2.8.pwconv1.weight":"convnext.encoder.stages.2.layers.8.pwconv1.weight",
"stages.2.8.pwconv1.bias":"convnext.encoder.stages.2.layers.8.pwconv1.bias",
"stages.2.8.pwconv2.weight":"convnext.encoder.stages.2.layers.8.pwconv2.weight",
"stages.2.8.pwconv2.bias":"convnext.encoder.stages.2.layers.8.pwconv2.bias",
"stages.3.0.gamma":"convnext.encoder.stages.3.layers.0.layer_scale_parameter",
"stages.3.0.dwconv.weight":"convnext.encoder.stages.3.layers.0.dwconv.weight",
"stages.3.0.dwconv.bias":"convnext.encoder.stages.3.layers.0.dwconv.bias",
"stages.3.0.norm.weight":"convnext.encoder.stages.3.layers.0.layernorm.weight",
"stages.3.0.norm.bias":"convnext.encoder.stages.3.layers.0.layernorm.bias",
"stages.3.0.pwconv1.weight":"convnext.encoder.stages.3.layers.0.pwconv1.weight",
"stages.3.0.pwconv1.bias":"convnext.encoder.stages.3.layers.0.pwconv1.bias",
"stages.3.0.pwconv2.weight":"convnext.encoder.stages.3.layers.0.pwconv2.weight",
"stages.3.0.pwconv2.bias":"convnext.encoder.stages.3.layers.0.pwconv2.bias",
"stages.3.1.gamma":"convnext.encoder.stages.3.layers.1.layer_scale_parameter",
"stages.3.1.dwconv.weight":"convnext.encoder.stages.3.layers.1.dwconv.weight",
"stages.3.1.dwconv.bias":"convnext.encoder.stages.3.layers.1.dwconv.bias",
"stages.3.1.norm.weight":"convnext.encoder.stages.3.layers.1.layernorm.weight",
"stages.3.1.norm.bias":"convnext.encoder.stages.3.layers.1.layernorm.bias",
"stages.3.1.pwconv1.weight":"convnext.encoder.stages.3.layers.1.pwconv1.weight",
"stages.3.1.pwconv1.bias":"convnext.encoder.stages.3.layers.1.pwconv1.bias",
"stages.3.1.pwconv2.weight":"convnext.encoder.stages.3.layers.1.pwconv2.weight",
"stages.3.1.pwconv2.bias":"convnext.encoder.stages.3.layers.1.pwconv2.bias",
"stages.3.2.gamma":"convnext.encoder.stages.3.layers.2.layer_scale_parameter",
"stages.3.2.dwconv.weight":"convnext.encoder.stages.3.layers.2.dwconv.weight",
"stages.3.2.dwconv.bias":"convnext.encoder.stages.3.layers.2.dwconv.bias",
"stages.3.2.norm.weight":"convnext.encoder.stages.3.layers.2.layernorm.weight",
"stages.3.2.norm.bias":"convnext.encoder.stages.3.layers.2.layernorm.bias",
"stages.3.2.pwconv1.weight":"convnext.encoder.stages.3.layers.2.pwconv1.weight",
"stages.3.2.pwconv1.bias":"convnext.encoder.stages.3.layers.2.pwconv1.bias",
"stages.3.2.pwconv2.weight":"convnext.encoder.stages.3.layers.2.pwconv2.weight",
"stages.3.2.pwconv2.bias":"convnext.encoder.stages.3.layers.2.pwconv2.bias",
"norm.weight":"convnext.layernorm.weight",
"norm.bias":"convnext.layernorm.bias",
"head.weight":"classifier.weight",
"head.bias":"classifier.bias"}

def run_test():
    for key in FB_TO_HF_MAP.keys():
        assert fb_to_hf_name(key) == FB_TO_HF_MAP[key], f"Not matching for {key}. {fb_to_hf_name(key)} != {FB_TO_HF_MAP[key]}"
    print("Pass: all tiny names match")

if __name__ == "__main__":
    run_test()