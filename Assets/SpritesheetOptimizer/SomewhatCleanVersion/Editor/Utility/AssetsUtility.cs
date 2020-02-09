using UnityEngine;
using UnityEditor;

public static class AssetsUtility
{
    public static bool FindAsset<T>(string assetName, out T asset) where T : Object
    {
        asset = default;
        var assetsWithTheSameName = AssetDatabase.FindAssets(assetName);
        if (assetsWithTheSameName.Length == 0)
            return false;
        for (int i = 0; i < assetsWithTheSameName.Length; i++)
        {
            var currentAssetPath = AssetDatabase.GUIDToAssetPath(assetsWithTheSameName[i]);
            var currentAsset = AssetDatabase.LoadAssetAtPath(currentAssetPath, typeof(Object));
            if (currentAsset.GetType().Equals(typeof(T)))
            {
                asset = (T)currentAsset;
                return true;
            }
        }
        return false;
    }

    public static void CreateAsAsset(this ScriptableObject scriptableObject, string path)
    {
        AssetDatabase.CreateAsset(scriptableObject, path);
        AssetDatabase.SaveAssets();
    }
}
