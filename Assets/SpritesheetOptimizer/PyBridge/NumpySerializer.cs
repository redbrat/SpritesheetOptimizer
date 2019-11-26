using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

public static class NumpySerializer
{
    private static Regex headerRegex = new Regex(@"{'descr': '([a-zA-Z0-9<]*)', 'fortran_order': (\w*), 'shape': ([0-9, ()]*),+", RegexOptions.Compiled);

    public static Array Deserialize(byte[] bytes)
    {
        if (bytes[0] != 147)
            throw new ArgumentException($"Bytes are not a numpy array.");

        var next5Characters = Encoding.ASCII.GetString(bytes, 1, 5);
        if (next5Characters != "NUMPY")
            throw new ArgumentException($"Bytes are not a numpy array.");

        var version = bytes[6];
        var subVersion = bytes[7];
        var headerLength = BitConverter.ToUInt16(bytes, 8);
        var header = Encoding.ASCII.GetString(bytes, 10, headerLength);
        var match = headerRegex.Match(header);
        if (!match.Success)
            throw new ArgumentException($"Header is not in right format: {header}");

        var descr = match.Groups[1].Value;
        var fortran_order = match.Groups[2].Value;
        var shape = match.Groups[3].Value;

        var shapeParts = shape.Substring(1, shape.Length - 2).Split(',').Select(s => Convert.ToInt32(s.Trim())).ToArray();

        var elementType = default(Type);
        var elementSize = default(int);
        var convertFunc = default(Func<byte[], int, object>);
        switch (descr)
        {
            case "<i4":
                elementType = typeof(Int32);
                elementSize = 4;
                convertFunc = (b, o) => BitConverter.ToInt32(b, o);
                break;
            case "u1":
                elementType = typeof(byte);
                elementSize = 1;
                convertFunc = (b, o) => b[o];
                break;
            default:
                throw new ArgumentException($"Unknown descr type: {descr}");
        }

        var offset = 10 + headerLength;
        var result = Array.CreateInstance(elementType, shapeParts);
        var indices = new int[shapeParts.Length];
        var length = shapeParts.Aggregate(1, (mul, i) => mul * i, mul => mul);
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < shapeParts.Length; j++)
            {
                var mulBefore = 1;
                for (int m = 0; m < j; m++)
                    mulBefore *= shapeParts[m];
                var mulAfter = 1;
                for (int m = j + 1; m < shapeParts.Length; m++)
                    mulAfter *= shapeParts[m];


                var currentIndexValue = default(int);
                if (j == 0)
                    currentIndexValue = i / mulAfter;
                else if (j == shapeParts.Length - 1)
                    currentIndexValue = i % shapeParts[j];
                else
                    currentIndexValue = i % (mulAfter * shapeParts[j]) / mulAfter;
                indices[j] = currentIndexValue;
            }
            result.SetValue(convertFunc(bytes, offset + elementSize * i), indices);
        }

        return result;
    }

    public static byte[] Serialize(Array array)
    {
        array = array.ConvertToMultiDimentional();

        var resultList = new List<byte>();

        resultList.Add(147);
        resultList.AddRange(Encoding.ASCII.GetBytes("NUMPY"));
        resultList.Add(1);
        resultList.Add(0);

        var descr = default(string);
        var toBytesFunc = default(Func<object, byte[]>);
        switch (array.GetType().GetElementType().FullName)
        {
            case "System.Int32":
                descr = "<i4";
                toBytesFunc = obj => BitConverter.GetBytes((int)obj);
                break;
            case "System.Byte":
                descr = "u1";
                toBytesFunc = obj => new byte[] { (byte)obj };
                break;
            default:
                throw new ArgumentException($"Unknown array type: {array.GetType().GetElementType().FullName}");
        }
        
        var fortran_order = $"False";
        var shape = new StringBuilder();
        var shapeParts = new int[array.Rank];
        for (int i = 0; i < array.Rank; i++)
        {
            if (i > 0)
                shape.Append(", ");
            shapeParts[i] = array.GetLength(i);
            shape.Append(shapeParts[i]);
        }

        var headerSb = new StringBuilder($"{{'descr': '{descr}', 'fortran_order': {fortran_order}, 'shape': ({shape.ToString()}), }}");
        while ((headerSb.Length + 10) % 64 != 63)
            headerSb.Append(' ');
        headerSb.Append('\n');
        var headerLength = headerSb.Length;

        resultList.AddRange(BitConverter.GetBytes((short)headerLength)); //9-10 байты
        resultList.AddRange(Encoding.ASCII.GetBytes(headerSb.ToString()));

        var length = array.Length;
        var indices = new int[shapeParts.Length];
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < shapeParts.Length; j++)
            {
                var mulBefore = 1;
                for (int m = 0; m < j; m++)
                    mulBefore *= shapeParts[m];
                var mulAfter = 1;
                for (int m = j + 1; m < shapeParts.Length; m++)
                    mulAfter *= shapeParts[m];

                var currentIndexValue = default(int);
                if (j == 0)
                    currentIndexValue = i / mulAfter;
                else if (j == shapeParts.Length - 1)
                    currentIndexValue = i % shapeParts[j];
                else
                    currentIndexValue = i % (mulAfter * shapeParts[j]) / mulAfter;
                indices[j] = currentIndexValue;
            }
            resultList.AddRange(toBytesFunc(array.GetValue(indices)));
        }

        return resultList.ToArray();
    }
}
