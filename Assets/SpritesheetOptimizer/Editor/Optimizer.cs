using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

public class Optimizer : EditorWindow
{
    private static Optimizer _intance;
    private static Sprite _sprite;
    private static Vector2Int _resolution;

    [MenuItem("Optimizer/Optimize")]
    private static void Main()
    {
        _intance = GetWindow<Optimizer>();
    }

    private static bool _displayingSomething;

    private void OnGUI()
    {
        var newSprite = EditorGUILayout.ObjectField(_sprite, typeof(Sprite), false) as Sprite;
        if (newSprite != _sprite)
            _sprite = newSprite;

        _resolution = EditorGUILayout.Vector2IntField("Area:" ,_resolution);

        if (_sprite != null && GUILayout.Button("Try") && !OptimizerAlgorythm.Working)
        {
            OptimizerAlgorythm.Go(_resolution, _sprite);
            _displayingSomething = true;
        }
        EditorGUILayout.LabelField(OptimizerAlgorythm.Working ? "Working" : "Idle");
        if (_displayingSomething)
        {
            EditorGUILayout.LabelField($"Op #{OptimizerAlgorythm.CurrentOp} of {OptimizerAlgorythm.CurrentOpsTotal} ops done");
            EditorGUILayout.LabelField($"Areas #{OptimizerAlgorythm.UniqueAreas} unique of {OptimizerAlgorythm.ProcessedAreas} processed areas");
            EditorGUILayout.LabelField($"Last pass lasted {OptimizerAlgorythm.LastPassTime} milliseconds or {OptimizerAlgorythm.LastPassTime / 1000f} seconds.");
            EditorGUILayout.Space();
            EditorGUILayout.LabelField($"Unoptimized pixels count: {OptimizerAlgorythm.UnoptimizedPixelsCount} of {OptimizerAlgorythm.OpaquePixelsTotal} total optimizable pixels");
        }

        Repaint();
    }
}
