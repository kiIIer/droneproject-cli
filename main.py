import argparse

import classifier
import presence_detector
import video_splitter


def main():
    parser = argparse.ArgumentParser(description='Video Processing Application')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    video_splitter.setup_cli(subparsers)
    presence_detector.setup_cli(subparsers)
    classifier.setup_cli(subparsers)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    # main()
    pass

import onnx


def check_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Check the model for internal consistency
    onnx.checker.check_model(model)

    # Print the ONNX model's metadata properties
    print("Model Metadata Properties:")
    for metadata in model.metadata_props:
        print(f"{metadata.key} : {metadata.value}")

    # Inspect the model's input shapes
    print("\nModel Inputs:")
    for input_tensor in model.graph.input:
        input_name = input_tensor.name
        # Get the type of the input tensor
        tensor_type = input_tensor.type.tensor_type
        # Get the shape of the input tensor
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.dim_param == "") else dim.dim_param for dim in
                 tensor_type.shape.dim]
        print(f"{input_name}: {shape}")

    # Inspect the model's output shapes
    print("\nModel Outputs:")
    for output_tensor in model.graph.output:
        output_name = output_tensor.name
        # Get the type of the output tensor
        tensor_type = output_tensor.type.tensor_type
        # Get the shape of the output tensor
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.dim_param == "") else dim.dim_param for dim in
                 tensor_type.shape.dim]
        print(f"{output_name}: {shape}")


# Call the function with your ONNX model path
check_onnx_model('test-files/best.onnx')
