using System;
using System.Threading;
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

    private ProgressReport _progressReport;
    private CancellationTokenSource _cts;

    private void OnGUI()
    {
        var newSprite = EditorGUILayout.ObjectField(_sprite, typeof(Sprite), false) as Sprite;
        if (newSprite != _sprite)
            _sprite = newSprite;

        _resolution = EditorGUILayout.Vector2IntField("Area:", _resolution);

        if (_sprite != null && _progressReport == null && GUILayout.Button("Try"))
        {
            var algorithmBulder = new AlgorythmBuilder();
            var algorythm = algorithmBulder
                .AddSizingsConfigurator<DefaultSizingsConfigurator>()
                .AddScoreCounter<DefaultScoreCounter>()
                .SetAreaEnumerator<DefaultAreaEnumerator>()
                .Build(getColors(_sprite));
            _progressReport = algorythm.ProgressReport;
            _cts = new CancellationTokenSource();
            launch(algorythm);
        }
        if (_cts != null)
        {
            EditorGUILayout.LabelField($"Unoptimized pixels count: {_progressReport.OverallOpsLeft} total optimizable pixels");
            EditorGUILayout.LabelField($"Current operation: {_progressReport.OperationDescription}");
            EditorGUILayout.LabelField($"Progress: {_progressReport.OperationsDone} of {_progressReport.OperationsCount}");
            if (GUILayout.Button("Cancel"))
            {
                _cts.Cancel();
                _cts = null;
            }
        }

        Repaint();
    }

    private async void launch(Algorythm algorythm)
    {
        await algorythm.Initialize(_resolution, _cts.Token);
        await algorythm.Run();
        _progressReport = null;
        _cts = null;
    }

    private MyColor[][][] getColors(Sprite sprite)
    {
        var texture = sprite.texture;
        var path = AssetDatabase.GetAssetPath(sprite);
        var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        ti.isReadable = true;
        ti.SaveAndReimport();

        var spritesCount = ti.spritesheet.Length;
        var sprites = (MyColor[][][])null;

        if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        {
            sprites = new MyColor[1][][];

            var tex = sprite.texture;
            var colors = new MyColor[tex.width][];
            for (int x = 0; x < tex.width; x++)
            {
                colors[x] = new MyColor[tex.height];
                for (int y = 0; y < tex.height; y++)
                {
                    var color = texture.GetPixel(x, y);
                    colors[x][y] = new MyColor(
                        Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
                    );
                }
            }
            sprites[0] = colors;
        }
        else
        {
            sprites = new MyColor[spritesCount][][];

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];

                var xOrigin = Mathf.FloorToInt(currentSprite.rect.x);
                var yOrigin = Mathf.CeilToInt(currentSprite.rect.y);
                var width = Mathf.CeilToInt(currentSprite.rect.width);
                var height = Mathf.CeilToInt(currentSprite.rect.height);
                var currentColors = new MyColor[width][];
                for (int x = 0; x < width; x++)
                {
                    currentColors[x] = new MyColor[height];
                    for (int y = 0; y < height; y++)
                    {
                        var color = texture.GetPixel(xOrigin + x, yOrigin + y);
                        currentColors[x][y] = new MyColor(
                            Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)),
                            Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)),
                            Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)),
                            Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
                        );
                    }
                }
                sprites[i] = currentColors;
            }
        }

        return sprites;
    }
}
