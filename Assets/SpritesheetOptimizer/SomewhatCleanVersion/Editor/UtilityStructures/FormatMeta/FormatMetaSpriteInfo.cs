using System;
using UnityEngine;

[Serializable]
public class FormatMetaSpriteInfo
{
    public string PathToFile => _pathToFile;
    public string NameOfSprite => _nameOfSprite;

    [SerializeField]
    private string _pathToFile;
    [SerializeField]
    private string _nameOfSprite;

    public FormatMetaSpriteInfo(string pathToFile, string nameOfSprite)
    {
        _pathToFile = pathToFile;
        _nameOfSprite = nameOfSprite;
    }
}