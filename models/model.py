import torch
from models.model_interface_abc import ModelInterface
from models.lambda_model import LambdaModel
from models.networks import MODEL_DICT
from utils.enums import Phase

def define_model(config: dict[str, dict], phase: Phase):
    device = torch.device(config["General"].get("device") or "cpu")
    model_config = config["General"]["model"]

    # Use only the netG_A_config for inference
    netG_A_config = dict(model_config["netG_A_config"])  # make a copy
    print("netG_A_config keys:", netG_A_config.keys())
    model_name = netG_A_config.pop("name")

    print("model_name:", model_name)

    netG_A_config["phase"] = phase
    netG_A_config["MODEL_DICT"] = MODEL_DICT
    netG_A_config["inference"] = config["General"].get("inference")

    print("netG_A_config inference:", netG_A_config["inference"])


    if issubclass(MODEL_DICT[model_name], ModelInterface):
        model = MODEL_DICT[model_name](**netG_A_config).to(device, non_blocking=True)
    else:
        model = LambdaModel(model_name, **netG_A_config).to(device, non_blocking=True)
    # Set inference_mode attribute for downstream use
    model.inference_mode = config["General"].get("inference", "netG_A")
    return model

# def define_model(config: dict[str, dict], phase: Phase):
#     device = torch.device(config["General"].get("device") or "cpu")
#     model_params: dict = config["General"]["model"]
#     model_name = model_params.pop("name")
#     model_params["phase"]=phase
#     model_params["MODEL_DICT"]=MODEL_DICT
#     model_params["inference"] = config["General"].get("inference")
#     if issubclass(MODEL_DICT[model_name], ModelInterface):
#         model = MODEL_DICT[model_name](**model_params).to(device, non_blocking=True)
#     else:
#         model = LambdaModel(model_name,**model_params).to(device, non_blocking=True)
#     return model