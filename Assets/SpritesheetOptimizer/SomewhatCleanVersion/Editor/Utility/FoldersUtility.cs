using System.IO;
using UnityEditor;

public static class FoldersUtility
{
    public static DefaultAsset GetOfCreateSubFolder(this DefaultAsset parent, string folderName)
    {
        var targetFolder = AssetDatabase.LoadAssetAtPath(Path.Combine(AssetDatabase.GetAssetPath(parent), folderName), typeof(DefaultAsset)) as DefaultAsset;
        if (targetFolder == default)
            targetFolder = CreateSubFolder(parent, folderName);
        return targetFolder;
    }

    public static DefaultAsset CreateSubFolder(this DefaultAsset parent, string folderName)
    {
        var newFolderGuid = AssetDatabase.CreateFolder(AssetDatabase.GetAssetPath(parent), folderName);
        var newFolderPath = AssetDatabase.GUIDToAssetPath(newFolderGuid);
        //AssetDatabase.ImportAsset(newFolderPath);
        return (DefaultAsset)AssetDatabase.LoadAssetAtPath(newFolderPath, typeof(DefaultAsset));
    }

    public static string GetPathToFileName(this DefaultAsset parent, string fileName)
    {
        var folderPath = AssetDatabase.GetAssetPath(parent);
        return Path.Combine(folderPath, fileName);
    }
}
