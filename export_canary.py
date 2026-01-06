from nemo.collections.asr.models import EncDecMultiTaskModel

canary:EncDecMultiTaskModel = EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-180m-flash") # type: ignore

canary.export("canary/flash.onnx")