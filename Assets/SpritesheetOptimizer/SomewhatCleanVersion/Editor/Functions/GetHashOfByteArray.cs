using System;
using System.Security.Cryptography;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(GetHashOfByteArray), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(GetHashOfByteArray), order = 0)]
public class GetHashOfByteArray : FunctionBase1<byte[], string>
{
    public override string Invoke(byte[] byteArray)
    {
        var hash = default(string);
        using (var sha1 = new SHA1CryptoServiceProvider())
        {
            hash = Convert.ToBase64String(sha1.ComputeHash(byteArray));
        }
        return hash;
    }
}
