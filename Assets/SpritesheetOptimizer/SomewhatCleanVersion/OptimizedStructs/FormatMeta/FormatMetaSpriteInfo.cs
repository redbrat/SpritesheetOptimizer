using System;
using UnityEngine;

[Serializable]
public class FormatMetaSpriteInfo
{
    public string PathToFile => _pathToFile;
    public string NameOfSprite => _nameOfSprite;
    public Vector2 Pivot => _pivot;
    public Vector2Int Size => _size;

    [SerializeField]
    private string _pathToFile;
    [SerializeField]
    private string _nameOfSprite;
    [SerializeField]
    private Vector2 _pivot;
    [SerializeField]
    private Vector2Int _size;

    public FormatMetaSpriteInfo(string pathToFile, string nameOfSprite, Vector2 pivot, Vector2Int size)
    {
        _pathToFile = pathToFile;
        _nameOfSprite = nameOfSprite;
        _pivot = pivot;
        _size = size;
    }
}