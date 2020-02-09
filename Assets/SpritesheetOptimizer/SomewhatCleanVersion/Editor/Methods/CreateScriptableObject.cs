using UnityEditor;
using UnityEngine;

public class CreateScriptableObject : MethodBase2<ScriptableObject, string>
{
    public override void Invoke(ScriptableObject scriptableObject, string path)
    {
        AssetDatabase.CreateAsset(scriptableObject, path);
        AssetDatabase.SaveAssets();
    }
}
