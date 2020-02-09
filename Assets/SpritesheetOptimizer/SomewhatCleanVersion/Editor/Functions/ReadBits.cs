using System;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(ReadBits), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(ReadBits), order = 0)]
public class ReadBits : FunctionBase3<byte[], int, int, int>
{
    public override int Invoke(byte[] bytes, int bitOffset, int valueLength)
    {
        var byteIndex = bitOffset / 8;
        var bitIndex = bitOffset % 8;

        var (maxAvailableSpace, initialToIntFunction) = getMaxSpaceAndFunc(bytes.Length, byteIndex);

        var availableSpace = maxAvailableSpace - bitIndex;
        if (availableSpace >= valueLength)
            return (initialToIntFunction(bytes, byteIndex) >> bitIndex) & valueLength.GetMaxInt();
        else
        {
            var part1 = (initialToIntFunction(bytes, byteIndex) >> bitIndex) & availableSpace.GetMaxInt();
            byteIndex += maxAvailableSpace / 8;
            var (_, secondaryToIntFunction) = getMaxSpaceAndFunc(bytes.Length, byteIndex);
            var part2 = (secondaryToIntFunction(bytes, byteIndex) & (valueLength - availableSpace).GetMaxInt()) << availableSpace;
            return part1 | part2;
        }
    }

    private (int, Func<byte[], int, int>) getMaxSpaceAndFunc(int byteArrayLength, int byteIndex)
    {
        var difference = byteArrayLength - byteIndex;
        if (difference >= 4)
            return (32, GetInt32);
        else if (difference >= 2)
            return (16, GetInt16);
        else
            return (8, GetInt8);
    }

    private int GetInt32(byte[] bytes, int byteIndex) => BitConverter.ToInt32(bytes, byteIndex);
    private int GetInt16(byte[] bytes, int byteIndex) => BitConverter.ToInt16(bytes, byteIndex);
    private int GetInt8(byte[] bytes, int byteIndex) => bytes[byteIndex];
}
