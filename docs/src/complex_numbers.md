# Complex Number Support

When registering input fields and output fields, you can pass `complex_dtype=True` to enable simple complex tensor input and output support. For instance, `Tin(..).register_input_field(input_field, complex_dtype=True)`.

Now the complex tensor support is limited in that the representation of complex numbers is a barebone 2D vector, since Taichi has no official support on complex numbers yet.

This means although `stannum` provides some facilities to deal with complex tensor input and output, you have to define and do the operations on the proxy 2D vectors yourself.

In practice, we now have these limitations:

* The registered field with `complex_dtype=True` must be an appropriate `VectorField` or `ScalarField`

    * If it's `VectorField`, `n` should be `2`, like `v_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, 3, 4, 5))`
    * If it's a `ScalarField`, the last dimension of it should be `2`,
        like `field = ti.field(ti.f32, shape=(2,3,4,5,2))`
    * The above examples accept tensors of `dtype=torch.cfloat, shape=(2,3,4,5)`

* The semantic of complex numbers is not preserved in kernels, so you are manipulating regular fields, and as a consequence, you need to implement complex number operators yourself

      * Example:

    ```python
    @ti.kernel
    def element_wise_complex_mul(self):
      for i in self.complex_array0:
          # this is not complex number multiplication, but only a 2D vector element-wise multiplication
          self.complex_output_array[i] = self.complex_array0[i] * self.complex_array1[i] 
    ```

## 
