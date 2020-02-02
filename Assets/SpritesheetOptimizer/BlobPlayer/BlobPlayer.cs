using System;

/*
 * Ок, судя по всему, я не оставил или потерял где-то документацию к формату. Придется восстанавливать...
 * 
 * 16 бит - short Длина длин в битах (lengthsLength)
 * lengthsLength бит - длина координат атласа по оси х
 * lengthsLength бит - длина координат атласа по оси y
 * lengthsLength бит - длина значений ширины вхождений атласа
 * lengthsLength бит - длина значений высоты вхождений атласа
 * 
 * lengthsLength бит - длина индекса спрайта, которому принадлежит область
 * lengthsLength бит - длина координаты x области
 * lengthsLength бит - длина координаты y области
 * lengthsLength бит - длина оффсетов спрайтов по х (?)
 * lengthsLength бит - длина оффсетов спрайтов по y (?)
 * 
 * Дальше идут записи атласа - x, y, width и height с соответствующими длинами в таком порядке.
 * 
 * 16 бит - short Кол-во рендереров в пуле, нужное для воспроизведения данной секвенции
 * lengthsLength бит - кол-во кадров секвенции
 * 
 * Дальше идут FramesCount оффсетов - сдвиги картинок x и y
 * 
 * Потом идут в оптимизированном виде записи об оффсетах областей в общем виде index, x, y.
 * 
 * 
 * В общем-то, нормальный вполне формат. Только надо зарезервировать хотя бы байт. И еще я думаю, надо все-таки сохранять информацию не только о сдвигах, но и о ширине-высоте оригиналных спрайтов.
 */

public sealed class BlobPlayer
{
    public int AtlasLength => _atlas.Length;

    public readonly int ChunksCount;
    public readonly int FramesCount;

    private readonly (int x, int y, int width, int height)[] _atlas;
    private readonly Chunk[][] _chunksByFrame;

    public Chunk[] GetFrame(int index) => _chunksByFrame[index];
    public (int x, int y, int width, int height) GetAtlasEntry(int index) => _atlas[index];

    public BlobPlayer(byte[] blob)
    {
        var offset = 0;
        var lengthsLength = MyBitConverter.ToInt(blob, 16, offset);

        var atlasXLength = MyBitConverter.ToInt(blob, lengthsLength, offset += 16);
        var atlasYLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var atlasWidthLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var atlasHeightLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);

        var atlasIndexLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var xLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var yLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var offsetXLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);
        var offsetYLength = MyBitConverter.ToInt(blob, lengthsLength, offset += lengthsLength);

        //Парсим атлас
        var atlasLength = MyBitConverter.ToInt(blob, 16, offset += lengthsLength);
        offset += 16;
        _atlas = new (int x, int y, int width, int height)[atlasLength];
        for (int i = 0; i < atlasLength; i++)
        {
            var x = MyBitConverter.ToInt(blob, atlasXLength, offset);
            offset += atlasXLength;

            var y = MyBitConverter.ToInt(blob, atlasYLength, offset);
            offset += atlasYLength;

            var width = MyBitConverter.ToInt(blob, atlasWidthLength, offset);
            offset += atlasWidthLength;

            var height = MyBitConverter.ToInt(blob, atlasHeightLength, offset);
            offset += atlasHeightLength;

            _atlas[i] = (x, y, width, height);
        }

        ChunksCount = MyBitConverter.ToInt(blob, 16, offset);
        FramesCount = MyBitConverter.ToInt(blob, lengthsLength, offset += 16);
        offset += lengthsLength;

        //Парсим оффсеты
        var offsetsX = new int[FramesCount];
        var offsetsY = new int[FramesCount];
        for (int i = 0; i < FramesCount; i++)
        {
            offsetsX[i] = MyBitConverter.ToInt(blob, offsetXLength, offset);
            offsetsY[i] = MyBitConverter.ToInt(blob, offsetYLength, offset += offsetXLength);
            offset += offsetYLength;
        }

        //Парсим чанки
        _chunksByFrame = new Chunk[FramesCount][];
        for (int f = 0; f < FramesCount; f++)
            _chunksByFrame[f] = new Chunk[ChunksCount];

        var getAtlasCoordinatesFunc = (Func<int, (int x, int y, int width, int height)>)(index => _atlas[index]);

        for (int c = 0; c < ChunksCount; c++)
        {
            var atlasIndex = default(int);
            var x = default(int);
            var y = default(int);
            for (int f = 0; f < FramesCount; f++)
            {
                if (MyBitConverter.ToInt(blob, offset++) == 0)
                    continue;
                var newIndex = MyBitConverter.ToInt(blob, offset++);
                var newX = MyBitConverter.ToInt(blob, offset++);
                var newY = MyBitConverter.ToInt(blob, offset++);

                if (newIndex == 1)
                {
                    atlasIndex = MyBitConverter.ToInt(blob, atlasIndexLength, offset);
                    offset += atlasIndexLength;
                }
                if (newX == 1)
                {
                    x = MyBitConverter.ToInt(blob, xLength, offset)/* - offsetsX[f]*/;
                    offset += xLength;
                }
                if (newY == 1)
                {
                    y = MyBitConverter.ToInt(blob, yLength, offset)/* - offsetsY[f]*/;
                    offset += yLength;
                }
                _chunksByFrame[f][c] = new Chunk(getAtlasCoordinatesFunc, atlasIndex, x, y);
            }
        }
    }
}
