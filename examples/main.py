import torch
from torch_mlir.fx import export_and_import, OutputType


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}"
        )
    return torch.matmul(a, b)

# Wrap the function in an nn.Module, since export_and_import traces a Module/callable
class MatmulModule(torch.nn.Module):
    def forward(self, a, b):
        return matmul(a, b)

model = MatmulModule().eval()

# Example inputs used to trace the graph and infer shapes
a = torch.randn(2, 1, 4, 5)
b = torch.randn(1, 3, 5, 6)
mlir_module = export_and_import(
    model,
    a,
    b,
    output_type="linalg-on-tensors",  # or "torch", "stablehlo", "raw"
    func_name="matmul",
)


# Save MLIR to file
with open("model.mlir", "w") as f:
    f.write(str(mlir_module))

