from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerRegressor


def test_onnx_export_model_imports_exist():
    """Ensure that the model classes used in export_onnx.py actually exist and match signatures."""
    # TinyCNN
    model1 = TinyCNNRegressor(in_channels=21, width=16)
    assert model1 is not None

    # EventTubeletTransformer
    model2 = EventTubeletTransformerRegressor(in_channels=21)
    assert model2 is not None


def test_export_onnx_script_syntax():
    """Ensure the export_onnx script can be imported without NameErrors (e.g. missing json, np)."""
    import sys
    from pathlib import Path

    # Add scripts to path so we can import it
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.append(str(scripts_dir))

    import export_onnx

    assert hasattr(export_onnx, "export_to_onnx")
    assert hasattr(export_onnx, "main")

    import generate_split_statistics

    assert hasattr(generate_split_statistics, "main")
