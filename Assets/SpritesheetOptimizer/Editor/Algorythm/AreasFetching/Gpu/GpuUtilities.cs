using UnityEngine;

public static class GpuUtilities
{
    public static (uint x, uint y, uint z) GetKernelThreadGroupSizes(this ComputeShader shader, int kernelIndex)
    {
        shader.GetKernelThreadGroupSizes(kernelIndex, out uint x, out uint y, out uint z);
        return (x, y, z);
    }
}
