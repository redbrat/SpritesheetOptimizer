using System.IO;
using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(WriteAndImportBytes), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.MethodsName + "/" + nameof(WriteAndImportBytes), order = 0)]
public class WriteAndImportBytes : MethodBase2<byte[], string>
{
    [SerializeField]
    private GetPathByFolderAndName _getPathByFolderAndName;

    public override void Invoke(byte[] blob, string path)
    {
        File.WriteAllBytes(path, blob);
        AssetDatabase.ImportAsset(path);
    }
}
