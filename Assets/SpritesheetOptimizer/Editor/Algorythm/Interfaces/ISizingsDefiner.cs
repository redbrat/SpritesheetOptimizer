using System.Collections.Generic;
using System.Threading;

public interface ISizingsConfigurator
{
    IEnumerable<MyVector2> ConfigureSizings(IEnumerable<MyVector2> result, int spritesCount, int xSize, int ySize, CancellationToken ct);
}