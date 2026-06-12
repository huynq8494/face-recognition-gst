import sys
from rknn.api import RKNN

if __name__ == '__main__':

    platform = "rk3588"
    do_quant = True

    # SCRFD convertion
    # model_path = "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx"
    # output_path = "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn"
    # dataset_path="/home/huynq/rknn_model_zoo/examples/test09/python/dataset/scrfd_dataset.txt"
    # rknn = RKNN(verbose=False)
    # rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[128, 128, 128]], target_platform=platform)
    # ret = rknn.load_onnx(model=model_path)
    # ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    # rknn.release()

    # Edgeface convertion

    model_path = "/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx"
    import onnx
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "dilations":
                    if len(attr.ints) != 2:
                        print(f"Fixing {node.name}: {attr.ints} -> [1,1]")
                        attr.ints[:] = [1, 1]
    onnx.save(model, "edgeface_xs_gamma_06_modified.onnx")

    model_path = "/home/huynq/models/edgeface/edgeface_xs_gamma_06_modified.onnx
    output_path = "/home/huynq/models/edgeface/edgeface_xs_gamma_06.rknn"
    dataset_path="/home/huynq/rknn_model_zoo/examples/test09/python/dataset/edgeface_dataset.txt"

    rknn = RKNN(verbose=False)
    # rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], target_platform=platform)
    rknn.config(target_platform=platform)
    ret = rknn.load_onnx(model=model_path, inputs=['input'], input_size_list=[[3, 112, 112]])
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    rknn.release()
