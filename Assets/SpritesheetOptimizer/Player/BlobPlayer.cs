using System;

public sealed class BlobPlayer
{
    private readonly (int x, int y, int width, int height)[] _atlas;
    private readonly Chunk[][] _chunksByFrame;

    public Chunk[] GetFrame(int index) => _chunksByFrame[index];

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

        //Парсим чанки
        var chunksCount = MyBitConverter.ToInt(blob, 16, offset);
        var framesCount = MyBitConverter.ToInt(blob, lengthsLength, offset += 16);
        offset += lengthsLength;

        _chunksByFrame = new Chunk[framesCount][];
        for (int f = 0; f < framesCount; f++)
            _chunksByFrame[f] = new Chunk[chunksCount];

        var getAtlasCoordinatesFunc = (Func<int, (int x, int y, int width, int height)>)(index => _atlas[index]);

        for (int c = 0; c < chunksCount; c++)
        {
            var atlasIndex = default(int);
            var x = default(int);
            var y = default(int);
            for (int f = 0; f < framesCount; f++)
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
                    x = MyBitConverter.ToInt(blob, xLength, offset);
                    offset += xLength;
                }
                if (newY == 1)
                {
                    y = MyBitConverter.ToInt(blob, yLength, offset);
                    offset += yLength;
                }
                _chunksByFrame[f][c] = new Chunk(getAtlasCoordinatesFunc, atlasIndex, x, y);
            }
        }
    }
}
