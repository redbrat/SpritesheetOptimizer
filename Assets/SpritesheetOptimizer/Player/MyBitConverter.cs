internal static class MyBitConverter
{
    internal static int ToInt(byte[] bytes, int bitsLength, int bitsOffset)
    {
        var result = 0;
        var end = bitsOffset + bitsLength;
        var localIndex = 0;
        for (int i = bitsOffset; i < end; i++)
            result |= ToInt(bytes, i) << localIndex++;
            //result |= bytes[i / 8] >> (i % 8) << localIndex++;
        return result;
    }

    internal static int ToInt(byte[] bytes, int bitsOffset) => bytes[bitsOffset / 8] >> (bitsOffset % 8);
}
