public class CpuBugTest
{
    public Algorythm.taskStruct[] TasksBuffer;
    public Algorythm.registryStruct[] RegistryBuffer; //Регистр картинок
    public Algorythm.areaStruct[] AreasBuffer;
    public int[] DataBuffer; //Цвета спрайтов
    public int[] CountsBuffer; //Count, 0 - это пустая область или повтор, в общем надо игнорить.
    public int[] ScoresBuffer; //Счет области - зачем считать на проце?

    public int MaxOpsAllowed;
    public int SpritesCount; //По идее можно, наверное, взять из RegistryBuffer, но, даже если можно, быстрее закешировать.

    (int, int, int) processSprite(int idx, int spriteIndex, Algorythm.registryStruct spriteInfo, int currentOpsCount, int areaSpriteOffset, int areaSpriteHeight, int areaX, int areaY, int areaWidth, int areaHeight)
    {
        int count = 0;
        int opsCountResult = currentOpsCount;
        int weAreReapeatingResult = 0; //Флаг обозначающий, что мы - повтор

        int spriteOffset = spriteInfo.SpritesDataOffset;
        int spriteWidth = spriteInfo.WidthAndHeight >> 16 & 65535;
        int spriteHeight = spriteInfo.WidthAndHeight & 65535;

        int spriteLastX = spriteWidth - areaWidth;
        int spriteLastY = spriteHeight - areaHeight;

        var task = TasksBuffer[idx];
        int currentSpriteX = task.SpriteXAndY >> 16 & 65535;
        int currentSpriteY = task.SpriteXAndY & 65535;
        int currentAreaX = task.XAndY >> 16 & 65535;
        int currentAreaY = task.XAndY & 65535;

        for (int spriteX = currentSpriteX; spriteX < spriteLastX; spriteX++)
        {
            for (int spriteY = currentSpriteY; spriteY < spriteLastY; spriteY++)
            {
                bool maybeThis = true;

                for (int x = currentAreaX; x < areaWidth; x++)
                {
                    for (int y = currentAreaY; y < areaHeight; y++)
                    {
                        opsCountResult++;

                        //Мы не закончили с этим спрайтом, но у нас уже не осталось доступных операций, так что сохраняем снапшот состояния и выходим из цикла.
                        if (opsCountResult > MaxOpsAllowed)
                        {
                            task.MetaAndSpriteIndex = 1 << 24 | spriteIndex;
                            task.SpriteXAndY = spriteX << 16 | spriteY;
                            task.XAndY = x << 16 | y;

                            break;
                        }
                        int sampleX = areaX + x;
                        int sampleY = areaY + y;
                        int candidateX = spriteX + x;
                        int candidateY = spriteY + y;

                        int sampleColor = DataBuffer[areaSpriteOffset + sampleX * areaSpriteHeight + sampleY];
                        int candidateColor = DataBuffer[spriteOffset + candidateX * spriteHeight + candidateY];

                        var sampleR = sampleColor >> 24 & 255;
                        var sampleG = sampleColor >> 16 & 255;
                        var sampleB = sampleColor >> 8 & 255;
                        var sampleA = sampleColor & 255;
                        var candidateR = candidateColor >> 24 & 255;
                        var candidateG = candidateColor >> 16 & 255;
                        var candidateB = candidateColor >> 8 & 255;
                        var candidateA = candidateColor & 255;

                        if (sampleColor != candidateColor)
                        {
                            maybeThis = false;
                            break;
                        }
                    }

                    currentAreaY = 0;

                    if (opsCountResult > MaxOpsAllowed)
                        break;
                    if (!maybeThis)
                        break;
                }

                currentAreaX = 0;

                if (opsCountResult > MaxOpsAllowed)
                    break;
                else
                if (maybeThis)
                {
                    if (task.AreasCounter >= idx)
                        count++;
                    else
                    {
                        weAreReapeatingResult = 1;
                        break;
                    }
                }

                task.AreasCounter++;
            }

            currentSpriteY = 0;

            if (opsCountResult > MaxOpsAllowed || weAreReapeatingResult == 1)
                break;
        }

        //Если закончили этот спрайт и у нас еще остались операции в запасе - мы можем работать со следующим спрайтом (если он есть (мы не знаем)), начиная с нулевых координат.
        if (opsCountResult <= MaxOpsAllowed)
        {
            task.SpriteXAndY = 0;
            task.XAndY = 0;
        }

        TasksBuffer[idx] = task;

        var resultVector = ( count, opsCountResult, weAreReapeatingResult );
        /*resultVector[0] = count;
        resultVector[1] = opsCountResult;
        resultVector[2] = weAreReapeatingResult;*/
        return resultVector;
    }

    //Если все работает как я думаю, то детали диспатча не важны. Нужно просто давать idx по порядку.
    public void Dispatch(int idx) 
    {
        var area = AreasBuffer[idx];

        int meta = area.MetaAndSpriteIndex >> 24 & 255;
        if (meta == 0) //Мета области равна 0, если она пустая или если она - повтор.
        {
            //CountsBuffer[id.x] = 1;
            return;
        }

        var formerTask = TasksBuffer[idx];

        int taskMeta = formerTask.MetaAndSpriteIndex >> 24 & 255;
        if (taskMeta == 0) //Мета задачи - это про то, закончили мы ее обсчитывать или нет: если закончили - выходим.
        {
            //CountsBuffer[id.x] = 2;
            return;
        }

        int areaSpriteIndex = area.MetaAndSpriteIndex & 16777215;
         
        int areaX = area.XAndY >> 16 & 65535;
        int areaY = area.XAndY & 65535;
        int areaWidth = area.WidthAndHeight >> 16 & 65535;
        int areaHeight = area.WidthAndHeight & 65535;

        if ((areaSpriteIndex == 2 || areaSpriteIndex == 0) && areaX == 0 && areaY == 0 && areaWidth == 8 && areaHeight == 8)
        { 

        } 

        var areaSpriteInfo = RegistryBuffer[areaSpriteIndex];
        int areaSpriteOffset = areaSpriteInfo.SpritesDataOffset;
        int areaSpriteWidth = areaSpriteInfo.WidthAndHeight >> 16 & 65535;
        int areaSpriteHeight = areaSpriteInfo.WidthAndHeight & 65535;

        int singleScore = ScoresBuffer[idx];
        if (singleScore == 0) //Если счет равен 0, возможен только 1 вариант - это первый проход и мы еще ни разу не считали счет. Потому что если считали он либо не 0 либо мы давно вышли по мете области
        {
            int opaquePixels = 0;
            for (int x = 0; x < areaWidth; x++)
            {
                for (int y = 0; y < areaHeight; y++)
                {
                    int xx = x + areaX;
                    int yy = y + areaY;
                    int color = DataBuffer[areaSpriteOffset + xx * areaSpriteHeight + yy];
                    int a = color & 255;
                    if (a != 0)
                        opaquePixels++;
                }
            }

            float square = areaWidth * areaHeight;
            singleScore = (int)(opaquePixels * opaquePixels * opaquePixels / square);
            if (singleScore == 0)
            {
                area.MetaAndSpriteIndex = 0 << 24 | areaSpriteIndex; //Область пустая - помечаем это в мете области
                formerTask.MetaAndSpriteIndex = 0 << 24 | 0;    //Если область пустая, мы должны как-то дать понять внешнему миру, что данный тред закончил работу. Мы могли бы 
                                                                //после каждого диспатча каждый раз забирать буффер областей и проверять если данная область пустая, или повторная, 
                                                                //а не только то, что мы с ней закончили, но это целый GetData ради 1 бита информации. Лучше уже пусть будет, что,
                                                                //если область пустая или повторная, это то же самое как если бы мы с ней закончили работать. Думаю, это даже и 
                                                                //логично.
                AreasBuffer[idx] = area;
                TasksBuffer[idx] = formerTask;
                //CountsBuffer[id.x] = 3;
                return;
            }
            //Получили счет области - сохраняем
            ScoresBuffer[idx] = singleScore;
        }

        //Ок, теперь мы проходимся по всем областям и смотрим уникальные ли они и считаем их кол-во

        int coincidentsCount = 0;
        int currentSpriteIndex = formerTask.MetaAndSpriteIndex & 16777215;

        int currentOpsCount = 0;

        for (int i = currentSpriteIndex; i < SpritesCount; i++)
        {
            var currentSpriteInfo = RegistryBuffer[i];

            (int, int, int) result = processSprite(idx, i, currentSpriteInfo, currentOpsCount, areaSpriteOffset, areaSpriteHeight, areaX, areaY, areaWidth, areaHeight);

            coincidentsCount += result.Item1;
            currentOpsCount = result.Item2;
            int weAreRepeating = result.Item3;

            if (weAreRepeating == 1)
            {
                //Тут тоже - если мы повторяемся, то надо заодно еще и таск пометить как завершенный.
                area.MetaAndSpriteIndex = 0 << 24 & areaSpriteIndex;
                formerTask.MetaAndSpriteIndex = 0 << 24 | 0;
                AreasBuffer[idx] = area;
                TasksBuffer[idx] = formerTask;
                ScoresBuffer[idx] = 0; //Даже если там не 0, у повторов должно быть 0 как флаг пропустить эту область.
                return; //Завершать незачем - выходим.
            }
            if (currentOpsCount > MaxOpsAllowed)
                break;
        }

        //Мы закончили со всеми спрайтами и уложились в доступное кол-во операций (может быть, впритык), так что можем сообщать, что закончили.
        if (currentOpsCount <= MaxOpsAllowed)
        {
            formerTask = TasksBuffer[idx];
            formerTask.MetaAndSpriteIndex = 0 << 24 | 0;
            TasksBuffer[idx] = formerTask;
            ////Тут тоже - если мы закончили, то надо заодно еще и таск пометить как завершенный.
            //area.MetaAndSpriteIndex = 0 << 24 & areaSpriteIndex;
            //AreasBuffer[id.x] = area;
        }

        //CountsBuffer[id.x] = 4;
        CountsBuffer[idx] = coincidentsCount;
    }
}
