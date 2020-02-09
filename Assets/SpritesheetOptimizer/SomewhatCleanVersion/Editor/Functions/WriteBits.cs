using UnityEngine;

[CreateAssetMenu(fileName = nameof(WriteBits), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(WriteBits), order = 0)]
public class WriteBits : FunctionBase4<int[], int, int, int, int>
{
    public override int Invoke(int[] buffer, int bitIndex, int value, int valueFrameLength)
    {
        int bufferIndex = bitIndex / 32;
        int startBitIndex = bitIndex % 32;
        int availableSpace = 32 - startBitIndex;
        if (availableSpace < valueFrameLength)
        {
            int firstChunk = (value & availableSpace.GetMaxInt()) << startBitIndex;
            int secondChunk = value >> availableSpace;

            buffer[bufferIndex] |= firstChunk;
            buffer[bufferIndex + 1] = secondChunk;
        }
        else
        {
            int theOneChunk = value << startBitIndex;
            buffer[bufferIndex] |= theOneChunk;
        }
        return bitIndex + valueFrameLength;
    }
}
