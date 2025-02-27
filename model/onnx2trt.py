import tensorrt as trt
import os
import sys
import onnx
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)


class TestOnnxModel:
     @staticmethod
     def checkup_model(model:str):
          logger.info("[info] start checking onnx model")
          try:
               onnx_model = onnx.load(model)
               onnx.checker.check_model(onnx_model)
               logger.info("[info] model is valid")
          except Exception as e:
               logger.error(f"[error] model is corrupted {e}")
               exit()

class LoggerTrt:
    @staticmethod
    def create_logger(flag_verbose:bool) -> trt.Logger:
        if flag_verbose:
            TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        else:
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        return TRT_LOGGER
    
class Builder:
    @staticmethod
    def build_engine(onnx_file_path: str, h:int, w:int,core_num:int,flag_verbose:bool, memmory_limit_gb:int = 1):
        TestOnnxModel.checkup_model(onnx_file_path)
        loggerTrt = LoggerTrt.create_logger(flag_verbose)
        logger.info("[info] Trt logger created")
        builder = trt.Builder(loggerTrt)
        logger.info("[info] Trt Builder created")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, loggerTrt)
        parser.parse_from_file(onnx_file_path)
        logger.info("[info] Onnx was parsed successfully")
        # Build the TensorRT engine
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("[info] Add flags to builder config")
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, (1, 3, h, w), (1, 3, h, w), (1, 3, h, w))
        config.add_optimization_profile(profile)
        logger.info("[info] Add optimization profile to builder config")

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,  memmory_limit_gb << 30)
        logger.info("[info] Set memmory pool limit")

        if core_num > 0 and builder.num_DLA_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.set_device_type(trt.DeviceType.DLA)
            config.DLA_core = core_num
            logger.info(f"[info] Install count of using DLA cores: {core_num} cores")

        engine = builder.build_serialized_network(network, config)
        if onnx_file_path.count(".") == 1:
            output_name:str = onnx_file_path.split(".")[0] + ".engine"
            with open(output_name, "wb") as f:
                f.write(engine)
        elif onnx_file_path.count(".") == 2:
            output_name:str = onnx_file_path.split(".")[1] + ".engine"
            with open(output_name, "wb") as f:
                f.write(engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output (for debugging)')
    parser.add_argument("--onnx-file", default="model.onnx",
                        help="path to onnx file")
    parser.add_argument('--dla-core', type=int, default=0,
                        help='id of DLA core for inference (0 ~ N-1)')
    parser.add_argument('--img', type=int, nargs='+', 
                        help='inference image size width, height')
    args = parser.parse_args()

    Builder.build_engine(args.onnx_file, args.img[0], args.img[1], args.dla_core, args.verbose)

