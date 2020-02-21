using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

[Serializable]
public class MainWindow : EditorWindow
{
    private static MainWindow _intance;
    private Convergence __convergence;
    private Convergence _convergence
    {
        get
        {
            if (__convergence == default)
            {
                var convergenceName = default(StringValue);
                if (!AssetsUtility.FindAsset("ConvergenceName", out convergenceName))
                    throw new ApplicationException("Can't find ConvergenceName asset! Try to reimport the asset from asset store.");
                var convergence = default(Convergence);
                if (!AssetsUtility.FindAsset(convergenceName.Value, out convergence))
                    throw new ApplicationException("Can't find Convergence asset! Try to reimport the asset from asset store.");
                __convergence = convergence;
            }
            return __convergence;
        }
    }

    private static Sprite _selectedSprite;

    private static string _atlasFolderName = "Assets/Atlas";
    private static TextAsset _importedFromCuda;

    private static Import _currentImport;

    [MenuItem("Optimizer/Main Window")]
    private static void windowInitializer() => _intance = GetWindow<MainWindow>();

    [MenuItem("Optimizer/Test")]
    private static void test()
    {
        var tex = new Texture2D(1, 1, TextureFormat.ARGB32, false, false);
        tex.filterMode = FilterMode.Point;
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();
        var times = 10_000;
        var arr = new Sprite[times];
        var before = GC.GetTotalMemory(true);
        for (int i = 0; i < times; i++)
            arr[i] = Sprite.Create(tex, new Rect(Vector2.zero, Vector2.one), Vector2.zero, 100, 0, SpriteMeshType.Tight, Vector4.zero, false);
        var after = GC.GetTotalMemory(true);
        var consumed = after - before;
        Debug.Log($"Roughly estimated size of sprite on managed side is {(consumed / (float)times).ToString("0.00")} bytes");

        DestroyImmediate(tex);
        for (int i = 0; i < times; i++)
            DestroyImmediate(arr[i]);
    }


    public class noSprite
    {
        public noSprite n;
    }
    public class yesSprite
    {
        public noSprite n;
        public Sprite s;
    }

    [MenuItem("Optimizer/Test 2")]
    private static void test2()
    {
        var times = 1_000_000;
        var yArr = new yesSprite[times];
        var nArr = new noSprite[times];
        var before = GC.GetTotalMemory(true);
        for (int i = 0; i < times; i++)
            nArr[i] = new noSprite();
        var after = GC.GetTotalMemory(true);
        var sizeOfNArr = after - before;
        before = GC.GetTotalMemory(true);
        for (int i = 0; i < times; i++)
            yArr[i] = new yesSprite();
        after = GC.GetTotalMemory(true);
        var sizeOfYArr = after - before;
        Debug.Log($"Size of reference to sprite is {((sizeOfYArr - sizeOfNArr) / (float)times).ToString("n2")}");
    }

    private void OnGUI()
    {
        _selectedSprite = EditorGUILayout.ObjectField("Sprite: ", _selectedSprite, typeof(Sprite), false) as Sprite;
        if (_selectedSprite != null && GUILayout.Button("Send to CUDA"))
            _convergence.ExportAllSprites(_selectedSprite);

        _atlasFolderName = EditorGUILayout.TextField("Atlas folder: ", _atlasFolderName);
        _importedFromCuda = EditorGUILayout.ObjectField("Imported from cuda file: ", _importedFromCuda, typeof(TextAsset), false) as TextAsset;
        if (_importedFromCuda != default && GUILayout.Button("Parse Cuda"))
            _currentImport = _convergence.Import(_importedFromCuda.bytes, _importedFromCuda.name);
        _currentImport = EditorGUILayout.ObjectField("Import: ", _currentImport, typeof(Import), false) as Import;
        if (_currentImport != default)
            ImportDrawer.Draw(_currentImport);
    }
}
