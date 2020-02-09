using System;
using UnityEngine;

[Serializable]
public class FormatMeta
{
    public FormatMetaSpriteInfo[] SpriteInfos => _spriteInfos;
    [SerializeField]
    private FormatMetaSpriteInfo[] _spriteInfos;

    public FormatMeta(FormatMetaSpriteInfo[] spriteInfos)
    {
        _spriteInfos = spriteInfos;
    }
}
