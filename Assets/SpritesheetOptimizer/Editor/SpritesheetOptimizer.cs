using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

public class SpritesheetOptimizerWindow : EditorWindow
{
    internal class SpritesheetChunk
    {
        private readonly Color[] _colors;

        public SpritesheetChunk(params Color[] args)
        {
            _colors = args;
        }

        public override int GetHashCode()
        {
            var sum = 0;
            for (int i = 0; i < _colors.Length; i++)
                sum += _colors.GetHashCode();
            return sum;
        }
    }

    private static SpritesheetOptimizerWindow _window;

    private static string _rootFolder = "Assets/TestFolder";

    private static UnityOptimizedSpritesStructure _chunksInfo;

    private static Vector2Int _areaResolution = Vector2Int.one * 4;

    [MenuItem("AwesomeTools/SpritesheetOptimizer")]
    private static void Main()
    {
        _window = GetWindow<SpritesheetOptimizerWindow>();
    }

    private void OnGUI()
    {
        _areaResolution = EditorGUILayout.Vector2IntField("Choose area size:", _areaResolution);

        _chunksInfo = (UnityOptimizedSpritesStructure)EditorGUILayout.ObjectField("Chunks", _chunksInfo, typeof(UnityOptimizedSpritesStructure));

        _rootFolder = EditorGUILayout.TextField("Root folder:", _rootFolder);
        if (GUILayout.Button("Choose folder..."))
        {
            var newFolder = EditorUtility.OpenFolderPanel("Optimized animator controllers folder", "Assets", "");
            if (newFolder.Contains("Assets"))
                _rootFolder = $"Assets/{newFolder.Split(new string[] { "Assets" }, StringSplitOptions.None)[1]}";
        }
            //_rootFolder = $"Assets/{EditorUtility.OpenFolderPanel("Optimized animator controllers folder", "", "").Split(new string[] { "Assets" }, StringSplitOptions.None)[1]}";
        //_rootFolder = EditorUtility.OpenFolderPanel("Optimized animator controllers folder", "", "");
        if (_chunksInfo != default && GUILayout.Button("Do"))
            doAllControllers();
    }

    private static void doAllControllers()
    {
        //var allAnimatorControllers = AssetDatabase.FindAssets("t:AnimatorController", null);
        var allAnimatorControllers = GetAssetsOfType<AnimatorController>(".controller");

        Debug.Log($"animatorControllers fount: {allAnimatorControllers.Length}");

        foreach (var controller in allAnimatorControllers)
        {
            Debug.Log($"doing controller {controller.name}");
            var controllerFolder = Path.Combine(_rootFolder, controller.name);
            Directory.CreateDirectory(controllerFolder);
            AnimatorControllerDoer.Do(controller, controllerFolder, _chunksInfo);
        }
    }

    //private static void @do(AnimatorController controller, OptimizedControllerStructure structure, List<SpritesheetChunk> futureSpriteSheet)
    //{
    //    var controllerFolder = Path.Combine(_rootFolder, controller.name);
    //    Directory.CreateDirectory(controllerFolder);
    //    AnimatorControllerDoer.Do(controller, structure, futureSpriteSheet, controllerFolder);
    //}

    /// <summary>
    /// Used to get assets of a certain type and file extension from entire project
    /// </summary>
    /// <param name="type">The type to retrieve. eg typeof(GameObject).</param>
    /// <param name="fileExtension">The file extention the type uses eg ".prefab".</param>
    /// <returns>An Object array of assets.</returns>
    public static T[] GetAssetsOfType<T>(string fileExtension) where T : UnityEngine.Object
    {
        List<T> tempObjects = new List<T>();
        DirectoryInfo directory = new DirectoryInfo(Application.dataPath);
        FileInfo[] goFileInfo = directory.GetFiles("*" + fileExtension, SearchOption.AllDirectories);

        int i = 0; int goFileInfoLength = goFileInfo.Length;
        FileInfo tempGoFileInfo; string tempFilePath;
        T tempGO;
        for (; i < goFileInfoLength; i++)
        {
            tempGoFileInfo = goFileInfo[i];
            if (tempGoFileInfo == null)
                continue;

            tempFilePath = tempGoFileInfo.FullName;
            tempFilePath = tempFilePath.Replace(@"\", "/").Replace(Application.dataPath, "Assets");

            Debug.Log(tempFilePath + "\n" + Application.dataPath);

            tempGO = AssetDatabase.LoadAssetAtPath<T>(tempFilePath);
            if (tempGO == null)
            {
                Debug.LogWarning("Skipping Null");
                continue;
            }
            else if (!(tempGO is T))
            {
                Debug.LogWarning("Skipping " + tempGO.GetType().ToString());
                continue;
            }

            tempObjects.Add(tempGO);
        }

        return tempObjects.ToArray();
    }
}
