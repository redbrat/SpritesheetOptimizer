public struct RegistryStruct
{
    public int SpritesByteOffset; //С какой позиции в байтовых буфферах начинается данный спрайт
    public int WidthAndHeight;

    public RegistryStruct(int spritesByteOffset, int widthAndHeight)
    {
        SpritesByteOffset = spritesByteOffset;
        WidthAndHeight = widthAndHeight;
    }
}
