# Dynamic Dimension Calculation

We first see what “dimension” means in the context of `stannum` and then see how we can leverage `DimensionCalculator` to enable dynamic dimension calculation

## Dimension != Shape

In `stannum`, “dimension” is virtual while “shape” is concrete. In other words, dimensions can be either integers or values of the enum `DimEnum` (namely `AnyDim`, `BatchDim` and `MatchDim(dim_id)`) while shapes can only mean integers.

After one dimension is concretized, a shape is produced. For example, dimensions `(AnyDim, 10)` contain both an integer and a `DimEnum`. To concretized the dimensions, we need a matching tensor, say a tensor of shapes `(123, 10)`, then the dimensions are concretized as `(123, 10)` as `AngDim` matches `123`. The same goes with dimensions that contains `BatchDim` and `MatchDim(dim_id)` besides some unification that is more complicated.

> Sidenote for developers: 
>
> In [tube.py](../../../src/stannum/tube.py), concretize_dimensions_and_unify() does exactly such concretization and unification!

Of course, in this context, dimensions can be also shapes, which makes dimensions a superset of shapes, which makes dimensions more powerful.

## `DimensionCalculator`

To calculate dimensions dynamically, we provide an API, which is `DimensionCalculator`, an abstract class containing only one method. So, alternatively, you can provide a closure as duck  `DimensionCalculator` implementation.

> Wait, why do we need dimensions/shapes in the first place?
>
> `Tube` help you manage fields automatically. By “manage”, it automatically create and destroy fields. In field creation, it needs (concrete) shapes instead of (virtual) dimensions. At the mean time, we need some way to express batching dimension, matching dimension and don’t-care (any) dimension, so we have dimensions.

The specification of `DimensionCalculator` is as simple as below:

```python
class DimensionCalculator(ABC):
    """
    An interface for implementing a calculator that hints Tube how to construct fields
    """

    @abstractmethod
    def calc_dimension(self,
                       field_name: str,
                       input_dimensions: Dict[str, Tuple[DimOption, ...]],
                       input_tensor_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[DimOption, ...]:
        """
        Calculate dimensions for a output/intermediate field

        @param field_name: the name of the field for which the dimensions are calculated
        @param input_dimensions: the dict mapping names of input fields to input fields
        @param input_tensor_shapes: the dict mapping names of input fields to
        shapes of input tensors that correspond to input fields
        """
        pass
```

`field_name` is the name of an intermediate/output field for which dimensions are calculated. `input_dimensions` gives you all the information of dimensions of all input fields. `input_tensor_shapes` gives you all the information of shapes of input tensors, of which names are the names of input fields.

## Example: Convolution

In the case of convolution, the shapes and dimensions of a convolution kernel and an input (say an image) are needed to compute the shapes of output.

In [test_dynamic_shape.py](../../../tests/test_tube/test_dynamic_shape.py), there is a simple `DimensionCalculator` for 1D convolution, namely `D1ConvDC`. And you can see how to use it in the test cases.