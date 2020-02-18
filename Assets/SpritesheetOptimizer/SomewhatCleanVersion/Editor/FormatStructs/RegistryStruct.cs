public struct RegistryStruct
{
    public int SpritesByteOffset; //С какой позиции в байтовых буфферах начинается данный спрайт
    public int SpritesBitOffset; //С какой позиции в битовыйх буфферах начинается данный спрайт
    public int WidthAndHeight;

    public RegistryStruct(int spritesByteOffset, int spritesBitOffset, int widthAndHeight)
    {
        SpritesByteOffset = spritesByteOffset;
        SpritesBitOffset = spritesBitOffset;
        WidthAndHeight = widthAndHeight;
    }
}
