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
    private Convergence _convergence;

    private static Sprite _selectedSprite;

    private static string _atlasFolderName = "Assets/Atlas";
    private static TextAsset _importedFromCuda;

    private static Import _currentImport;

    [MenuItem("Optimizer/Main Window")]
    private static void windowInitializer()
    {
        var convergenceName = default(StringValue);
        if (!AssetsUtility.FindAsset("ConvergenceName", out convergenceName))
            throw new ApplicationException("Can't find ConvergenceName asset! Try to reimport the asset from asset store.");
        var convergence = default(Convergence);
        if (!AssetsUtility.FindAsset(convergenceName.Value, out convergence))
            throw new ApplicationException("Can't find Convergence asset! Try to reimport the asset from asset store.");

        _intance = GetWindow<MainWindow>();
        _intance._convergence = convergence;
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
