using System;

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
