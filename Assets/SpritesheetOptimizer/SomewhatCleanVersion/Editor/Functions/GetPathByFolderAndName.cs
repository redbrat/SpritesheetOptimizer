using System.IO;
using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(GetPathByFolderAndName), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(GetPathByFolderAndName), order = 0)]
public class GetPathByFolderAndName : FunctionBase2<DefaultAsset, string, string>
{
    public override string Invoke(DefaultAsset folder, string fileName)
    {
        var folderPath = AssetDatabase.GetAssetPath(folder);
        return Path.Combine(folderPath, fileName);
    }
}
