
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "file_reader.h"
#include "bit_converter.h"

using namespace std;

/*
Ок, это первый метод, который я параллелю, поэтому давайте решим, как я это буду делать. Просто по индексу блока мы не можем вычислить никакие индексы,
потому что ширина-высота спрайтов неодинакова. Точнее можем, но это будет очень неэффективно. А что мы должны сделать - это каким-то образом разделить
задачи по спрайтам так же как по сайзингам и инфу о сайзингах и размерах спрайтов закэшировать. Можно сказать так - кол-во блоков = кол-во сайзингов *
кол-во спрайтов. Таким образом, поток может легко вычислить спрайт и сайзинг, которые ему обрабатывать. Если его индекс 126, а кол-во сайзингов и спрайтов
 - 22 и 40 соответственно, то его сайзинг = 126 / 40, а спрайт - 126 % 40, т.е. сайзинг четвертый, спрайт - 6й. А всего блоков 22 * 40 = 880. Че-то мало.
Хотя, это только блоки. А как внутри потоки будут узнавать о своих конкретных назначенных пикспелях? А вот как. Мы будем запускать кернел для каждого
из 880 вариантов. Поэтому для каждого потока будет очевидно кто он и где он, т.к. будет передваваться ширина и высота. Иммет ли смысл делать столько
кернелов? Для среднего размера спрайтов, скажем 128х60 вся работа целого кернела будет состоять из всего нескольких сотен тысяч операций. Целого кернела,
Карл! Это если размер блока задать 128, то будет всего 8 блоков. И это для средних размеров спрайтов. Не маловато ли работы? Может быть имеет смысл сделать
опциональное разделение кернелов по спрайтам - 1 кернел == 1 спрайт. Тогда если размер блока будет 128, блоков будет уже по крайней мере в районе сотни.
Внутри такого блока мы на вход получаем смещение спрайта, ширину и высоту, а, будучи потоком, мы можем вычислить свою позицию стандартно, т.к. ширина и
высота тут постоянные. Возможно для уменьшения дайверженс и лучшей работы с памятью стоит изолировать также разные сайзинги по блокам. Т.е. если размер
блока будет, скажем, 128, сайзинг - 8х8, а спрайт 128х48, мы запускаем кернел, передавая ему оффсет спрайта в данных, 128 и 48. На уровне блоков мы действуем
следующим образом. У нас сайзинг 8х8, поэтому рабочая ширина у нас 120, а высота - 40. Таким образом нам нужно иметь 120 х 40 = 4800 потоков. Таким образом
на этом сайзинге у нас будут работать 4800 / 128 + 1 = 38 блоков. В статической памяти у них будет храниться вся нужная им инфа - их сайзинг, их референсная
область, возможно, если памяти хватит - их карты пустоты для данного спрайта. Вообще-то памяти как-то мало везде... Может использовать 1 блок на
сайзинг-спрайт, просто очень большой? Скажем, пусть размер блока будет максимальным (1024), тогда мы сможем грузить в шаренную память спрайт, если не
целиком, то координированно частями и пройтись по нему за 5 неполных итераций. Т.е. каждый поток будет ответственнен за 4-5 отдельный областей поочередно.
Вообще система хорошая: каждый поток имеет 4-5 отдельных заданий, спрайт, загруженный в шаред мемори и можно по частям проходиться по любого размера
спрайтам. Это, конечно, все прикольно, но тогда, получается, у нас по одному блоку на кернел? А, нет, стоп. По спрайту на кернел. А в кернеле кол-во блоков
будет равно кол-ву сайзингов. Т.е. на нашем случае будет 40 кернелов, по 22 блока в каждом, и у каждого блока по 1024 потока. Кол-во потоков - 901_120.
Каждый поток будет выполнять в среднем 4.5 * 120 * 40 = 21_600 операций с, в основном, шаред памятью. Пока что с моим текущим пониманием того, как это
работает, мне это кажется отличным вариантом. И кстати, регистр и сайзинги в любом случае должны с запасом влезть в константную память.

Стоп, я что-то не додумал... Сравнивать-то мне надо будет со ВСЕМИ спрайтам. Так что память будет использоваться не только шаред, но и общая. Ну уместить всю
дату в шаред мемори не представляется возможным... Хотя, если распределить задачи по времени... Чтобы в кернеле было не 22 блока а 40х22 = 880 блоков! В
каждом по 1024 потока. После завершения всех - тупо складываем результат по сайзингам, будет 22*120*40 результата на кернел как и надо. Результатов будет
много - 22*120*40*40, более 4 миллионов. Сложить их можно будет также параллельно - завершающим складывающим кернелом. На выходе он нам сделает наши 105_600
результатов - по 4800 (120х40) результатов на спрайт для каждого из 22х сайзингов. Со всех 40 кернелов получится обратно 4 миллиона результатов, из которых
мне самой элементарной сортировкой надо будет отобрать наибольший результат. Самой элементарной - это значит запускаем блоки по 1024 потока, агрессивными
оптимизациями в 10 проходов получаем 1 наибольший, пишем в буфер, и проводим с ним ту же операцию, пока не останется 1 значение. Шикарно.

А войдмапы пускай клиенты делают - это не особо их затруднит, я думаю.

P.S. Тут только проблемы с адресацией опять же возникнут, но я уже придумал как их решить. Составляем заранее (опять же - клиентом) мапу... Хотя нет, даже
этого не нужно. Каждый блок знает заранее с какого спрайта у него референс, а с какого претендер. Так что он просто сразу знает и оба оффсета. Только вот еще
одна проблема потенциальная - т.к. мы занимаем память, скорее всего, по максимуму, на одном мультипроцессоре будет у нас жить тоько 1 блок. А работать этот
блок будет с глобал мемори. Это значит большие задержки и простой мультипроцессоров. Решение здесь - ограничить сильнее использование шаред мемори - пусть
лучше будет больше итераций. Т.е. вообще-то кол-ву памяти будет достаточно быть равным кол-ву потоков - 1024 потокам не нужно больше инфы в шаред мемори, чем
1024 + полщадь сайзинга (8х8=64 в нашем случае). Если на каждый пиксель надо будет иметь 5 байтов (4 байта цвета + 1 байт пустоты), это всего лишь около 5.3кб
памяти на блок в одно и то же время. Это значит на одном мультипроцессоре смогут сожительствовать 18 блоков. Думаю, может быть норм, особенно учитывя еще, что
в 1 процессоре одновременно могут проходить только 4 варпа, но надо смотреть. Вообще круто, что на моей карточке 96Кб общей памяти - наибольшее кол-во из
текущих. Посмотрим, что принесет 8.0 compute capability... В общем, если будет не хватать памяти, буду уменьшать размер блока и соответсвенно размер требуемой
шаред мемори на блок, таким образом увеличивая кол-во блоков на процессор.


5.12.2019
Все-таки решил заполнять карты пустот самому. Ибо для больших наборов спрайтов (а скорее всего наборы будут большими) эти карты пустот будут большого размера.
Для 100 мегапикселей например, карта пустот с 22 сайзингами будет дополнительными 300 мегабайтами информации. Есть мнение (мое), что это быстрее обсчитается
на видеокарте, чем загрузится в память... Хотя все же нет, 300 мб не так много. Это во-первых. А во-вторых вычислительные мощности - дефицитный ресурс, узкое
место. Даже если все это и обсчитается быстрее, чем прочитается с диска и загрузится, шине и диску все равно делать нефига, а за то время пока они грузят,
карта обсчитает больше действителньо нужной инфы.

P.S. Я ошибся - войдмапы занимают меньше места: не в 3 раза больше для 22 сайзингов, а всего где-то 3/4 размера. Забыл, что в каждом пикселе еще по 4 байта.


6.12.2019
Ок, я что-то не подумал. Кернелы не могут выполняться по-настоящему параллельно. Поэтому надо делать все одним кернелом. Какие следствия из этого? Да
собственно не такие уж большие, раз я решил разбивать задачи по блокам не равномерно. Просто теперь blockIdx.x будет содержать трехмерным.

Ок, с адресацией решили. Но что теперь делать с дайверженси? Пожалуй уже сейчас пора принимать меры, потому что мы уже кешируем домен, а домен разным потокам 
нужен все больше и больше разный.

Итак, разберемся что же происходит у нас на уровне блока. По сути мы хотим предотвратить дайверженси разделив задачу на небольшие участки. Это значит, нам 
понадобится трекать потоки, которые уже закончили работу и регулярно проводить компрессию. В принципе можно даже ввести параметр - как часто мы хотим 
проводить компрессию на уровне блока и через него найти компромисс между оккупацией и производительностью.

Это все хорошо, но это все влечет за собой то, что потоки должны будут свичиться между задачами. Не слишком ли большой оверхед для возможности ограничить
дивергнецию? Да нет вроде. Тем более что не обязательно юзать шаред мемори для хранения контекстов. Хотя нет, нужно - шафл работает только на уровне ворпа. 
В контексте надо будет хранить координату, на которой мы остановились. В пределах 1 спрайта это вполне может занимать 1 байт. Реже 2 байта. А можно вообще 
просто присылать в кернел максимальное значение ширины/высоты и хранить побитно.

Так что если какой-то поток закончил обработку (нашел такую же область и посчитал себя повтором), то он прибавляет кол-во простаивающих потоков на 1. А во 
время следующей процедуры сжатия, он возмет себе контекст соседа и продолжит его работа с момента на котором тот остановился. Ах да, еще контект должен 
содержать порядковый номер области, чтобы поток знал какая область его. И все, вся инфа у него есть, пошел работать. А поток, чья работа ему досталась взял 
себе например новую область, которую пока еще никто себе не брал. Или опять же - область соседа, если есть такая.

Только я вот начал сомневаться в разумности загрузки домена в шаред-память. В конце-концов как узнать какую область нам грузить? Даже в самом простом случае 
это нетривиально, учитывая, что загрузить надо с запасом в сайзинг. Кто будет эти доп. данные грузить? А как быть в случае с пропусками? При большом разбросе 
есть вероятность, что нам понадобится весь кандидатский спрайт. Думаю, надо отказаться от идеи кеширования домена в шаред-мемори. Может это и к лучшему. 
Во-первых будет большая оккупация. Во-вторых можно будет не особо экономить на других вещах, скажем, не хранить данные контекста побитово, а тупо каждому 
выделить по 2 байта с барского плеча и пусть ни в чем себе не отказывают. Так 1 блок будет занимать 6Кб. Вообще-то все равно дофига, оккупация будет 
слабенькой. Лучше уже хранить побитово. Да уж, если контекст столько занимает, может и все равно пришлось бы отказаться от кеша, т.к. лучше обойтись без кеша, 
чем иметь большую дивергенцию. Получается, что для средних заданий, скажем изображения до 64х64 и до 256 спрайтов у нас контекст одного потока займет 20 
битов, а это оккупация 38 блоков на см! Большие же задания приведут к меньшей оккупации, но, т.к. там сами блоки будут выполняться дольше, думаю, это в 
какой-то мере скомпенсирует меньшее кол-во блоков на см.

А стоит ли вообще заморачиваться с контекстами? Вообще-то чем больше размеры спрайта тем более стоит и чем меньше тем менее. А раз мы ориентируемся на 
большие спрайты, то, видимо, стоит. Допустим, у нас задание, состоящее из спрайтов 256х192. Это (256 - 8) * (192 - 8) * 64 = 2_920_448 операций для каждого 
потока. Выигрыш может быть значительным.


7.12.19
Ок, оказалось, что лучше загружать в см не наш спрайт а спрайт кандидата, т.к. наш спрайт всегда разный у каждого блока и на карте более оптимизирована для 
массовых запросов глобальная память. Запрос же одного и того же значения разными потоками, который гораздо чаще будет осуществляться именно к спрайту 
кандидата, более оптимизирован из шаред мемори - там по сути идет 1 запрос и остальным потокам это значение раздается мультикастом. Я подумываю сделать такую 
оптимизацию - загружать в шаред мемори не всю инфу, а лишь по одному каналу. Т.к. нас интересуют совпадения, этой инфы хватит в 255 из 256 случаев, т.е. мы 
таким образом уменьшаем пол-во запросов к глобал мемори до 1/256 случая.

Таким образом использование шаред мемори увеличивается на 1024 байта, если мы кешируем только кандидатский спрайт и до 2 кб, если еще и свой. Хотя... 
Вообще-то да, там ведь из-за ужимания будет та же проблема с невозможностью предсказать наличие в шаред-мемори ни одного пикселя кроме своего. Да, ок. 
Получается, единственное что мы имеем возможность кешировать с таким подходом - это вспомогательные флаги, скажем, карты пустот, или может быть придумать 
еще какой-то признак, типа r >= 127, сразу уменьшая кол-во загрузок из глобальной памяти вдвое. Для большого спрайта, размером, скажем, 256х256, такая маска 
будет весить 8 кб, что вполне нормально.

Можно было бы конечно разделить спрайты на дальнейшие куски, таким образом, чтобы стало возможным загрузить всю инфу этого куска в шаред-память полностью и 
это не сильно сказалось бы на оккупанси. Давайте посчитаем - допустим приемлемое кол-во занимаемой памяти для блока - 16 кб, это 6 резидентов на процессоре. 
Все равно мало, но допустим. Тогда мы должны будем ограничиться куском в 4096 пикселей, это, например, картинка 64х64. И это только для одного спрайта. 
Загрузив два спрайта, пикселей будет всего 2048, или куски примерно 64х32. Получается, каждый поток сможет проверить лишь 2048 областей. А учитывая 
использование войдмапов и того меньше. В общем, не факт, что оно того стоит, непосредственная работа может быть сделается быстрее, чем загрузится такое 
кол-во памяти, т.е. налицо будет неравномерное распределение нагрузки по ресурсам видеокарты. Или, наоборот более равномерное? Вроде бы пока инфа грузится 
процессор может выполнять новые ворпы. Но в случае с 16 кб, если всего 5 альтернативных блоков по 32 ворпа, т.е. 160 альтернативных ворпов. Мне кажется, если 
уж делать нагружать память по полной, нужно обеспечить высокую оккупацию. Если сделать так, что блок будет работать с 4 кб инфы от своего и кандидата, то 
это по 8 кб на блок и 12 резидентов, или 384 альтернативных ворпа.

Так, так, нет, отмена. Максимальное кол-во варпов-резидентов на см составляет 64 для моей архитектуры, значит, в целом, для блока шириной 1024, оптимальное 
потребление шаред памяти - 48 кб. Тогда на см будет ровно два блока-резидента и вся память будет зайдествована. Что это значит? Это значит что мы не можем 
обеспечить высокую оккупацию за счет увеличения кол-ва блоков или ворпов (<=64) или потоков (<=2048) в принципе. Таким образом нам остается использовать эти 
48 кб на блок максимально эффективно, в том числе для уменьше дайерженси и увеличения оккупации своими силами. А это как раз, что достигается тем, о чем я 
рассуждал в позапрошлом абзаце. Но тогда я не знал про это ограничение. Теперь нам остается только решить на что потратить эти 48 кб.

Предположим, что средний размер спрайта у нас будет 256х256.
Тогда одна битовая маска у нас будет занимать 8кб. Мы можем взять, например:
1 карту пустоты полную.
4 битовых карты для 4х каналов.
Оставшиеся 8 кб - это 8 байтов информации на поток. Пока что из них я вижу 6 уйдет на контекст - 2 шорта для описания своей области и кандидатской и по 1 
байту на описание текущей точки остановки. Вообще, я могу выделить и по полтора байта, таким образом сразу предупредив поддержку спрайтов до 4096х4096. И 
таким образом у меня останется всего 1 неиспользованный байт.

*/


#define BLOCK_SIZE 1024
#define DIVERGENCE_CONTROL_CYCLE 128 //Через какое кол-во операций мы проверяем на необходимость ужатия, чтобы избежать дивергенции?
#define DIVERGENCE_CONTROL_THRESHOLD 32 //Если какое число потоков простаивают надо ужимать? Думаю это значение вряд ли когда-нибудь изменится.

__constant__ int SpritesCountBits;
__constant__ int MaxWidthBits;
__constant__ int MaxHeightBits;
__constant__ int ContextBits;
__constant__ int SpritesCountBitMask;
__constant__ int MaxWidthBitMask;
__constant__ int MaxHeightBitMask;

__global__ void mainKernel(int sizingsCount, short* sizingWidths, short* sizingHeights, int spritesCount, int* offsets, short* widths, short* heights, unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a)
{
	int ourSpriteIndex = blockIdx.x;
	int candidateSpriteIndex = blockIdx.y;
	int sizingIndex = blockIdx.z;

	int ourOffset = offsets[ourSpriteIndex];
	int ourWidth = widths[ourSpriteIndex];
	int ourHeight = heights[ourSpriteIndex];
	int ourSquare = ourWidth * ourHeight;

	int candidateSpriteOffset = offsets[candidateSpriteIndex];
	int candidateSpriteWidth = widths[candidateSpriteIndex];
	int candidateSpriteHeight = heights[candidateSpriteIndex];
	int candidateSpriteSquare = candidateSpriteWidth * candidateSpriteHeight;

	int sizingWidth = sizingWidths[sizingIndex];
	int sizingHeight = sizingHeights[sizingIndex];


	//if (threadIdx.x == 0)
	//	printf("Hello from block! My sprite is #%d (width %d, height %d) and I work with sprite %d (width %d, height %d) and sizing %d (width %d, height %d) \n", ourSpriteIndex, ourWidth, ourHeight, candidateSpriteIndex, candidateSpriteWidth, candidateSpriteHeight, sizingIndex, sizingWidth, sizingHeight);

	extern __shared__ int contexts[];

	int myInitialContextAddress = threadIdx.x * ContextBits;
	int myInitialContextAddressByteIndex = myInitialContextAddress / 8;
	int myInitialContextAddressBitOffset = myInitialContextAddressByteIndex % 8;

	int myArea = threadIdx.x;
	int currentCandidateX = 0;
	int currentCandidateY = 0;
}

__global__ void testKernel(int sizingsCount, short* sizingWidths, short* sizingHeights, int spritesCount, int* offsets, short* widths, short* heights, unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a)
{
	if (threadIdx.x == 0)
	{
		printf("%d\n", threadIdx.x);
		return;
	}
	int ourSpriteIndex = blockIdx.x;
	int candidateSpriteIndex = blockIdx.y;
	int sizingIndex = blockIdx.z;

	int ourOffset = offsets[ourSpriteIndex];
}

int getBitsRequired(int value)
{
	if (value < 2)
		return 1;
	if (value < 4)
		return 2;
	if (value < 8)
		return 3;
	if (value < 16)
		return 4;
	if (value < 32)
		return 5;
	if (value < 64)
		return 6;
	if (value < 128)
		return 7;
	if (value < 256)
		return 8;
	if (value < 512)
		return 9;
	if (value < 1024)
		return 10;
	if (value < 2048)
		return 11;
	if (value < 4096)
		return 12;
	if (value < 8192)
		return 13;
	if (value < 16384)
		return 14;
	if (value < 32768)
		return 15;
	if (value < 65536)
		return 16;
	return -1;
}

int main()
{
	string path = "P:\\U\\Some2DGame\\Cuda\\info\\data.bytes";
	tuple<char*, int> blobTuple = file_reader::readFile(path);
	char* blob = get<0>(blobTuple);
	int blobLength = get<1>(blobTuple);

	int metaLength;
	memcpy(&metaLength, blob, 4);
	int combinedDataOffset = metaLength + 4;

	short spritesCount;
	memcpy(&spritesCount, blob + combinedDataOffset + 2, 2);
	short sizingsBlobLength;
	memcpy(&sizingsBlobLength, blob + combinedDataOffset + 4, 2);

	short sizingsCount = sizingsBlobLength / 4;

	int registryStructureLength = 8;

	char* sizingsBlob = blob + combinedDataOffset + 6;
	char* registryBlob = sizingsBlob + sizingsBlobLength;
	int registryBlobLength = spritesCount * registryStructureLength;
	char* dataBlob = registryBlob + registryBlobLength;
	//int dataBlobLength = blobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;
	//char* dataBlobs = new char[spritesCount];

	int dataBlobLineLength = 0;
	int maxWidth = 0;
	int maxHeight = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		int width = bit_converter::GetShort(registryBlob, spritesCount * 4 + i * 2);
		int height = bit_converter::GetShort(registryBlob, spritesCount * 6 + i * 2);
		if (width > maxWidth)
			maxWidth = width;
		if (height > maxHeight)
			maxHeight = height;
		dataBlobLineLength += width * height;
	}

	int spritesCountBits = getBitsRequired(spritesCount);
	int maxWidthBits = getBitsRequired(maxWidth);
	int maxHeightBits = getBitsRequired(maxHeight);
	int contextBits = SpritesCountBits + MaxWidthBits + MaxHeightBits;
	cudaMemcpyToSymbol(&SpritesCountBits, &spritesCountBits, 4);
	cudaMemcpyToSymbol(&MaxWidthBits, &maxWidthBits, 4);
	cudaMemcpyToSymbol(&MaxHeightBits, &maxHeightBits, 4);
	cudaMemcpyToSymbol(&ContextBits, &contextBits, 4);

	int spritesCountBitMask = 1 << spritesCountBits - 1;
	int maxWidthBitMask = 1 << maxWidthBits - 1;
	int maxHeightBitMask = 1 << maxHeightBits - 1;
	cudaMemcpyToSymbol(&SpritesCountBitMask, &spritesCountBitMask, 4);
	cudaMemcpyToSymbol(&MaxWidthBitMask, &maxWidthBitMask, 4);
	cudaMemcpyToSymbol(&MaxHeightBitMask, &maxHeightBitMask, 4);

	int dataBlobLength = dataBlobLineLength * 4;
	//int voidsBlobLength = (dataBlobLength / 32 * sizingsCount) / 8 + 1;
	//int voidsBlobLength = blobLength - dataBlobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;
	int voidsBlobLength = bit_converter::GetInt(dataBlob + dataBlobLength, 0);
	char* voidsBlob = dataBlob + dataBlobLength + 4;


	int lineMaskLenght = bit_converter::GetInt(voidsBlob + voidsBlobLength, 0);

	char* rFlagsBlob = voidsBlob + voidsBlobLength + 4;
	char* gFlagsBlob = rFlagsBlob + lineMaskLenght;
	char* bFlagsBlob = gFlagsBlob + lineMaskLenght;
	char* aFlagsBlob = bFlagsBlob + lineMaskLenght;

	char* deviceSizingsPtr;
	cudaMalloc((void**)&deviceSizingsPtr, sizingsBlobLength);
	char* deviceRegistryPtr;
	cudaMalloc((void**)&deviceRegistryPtr, registryBlobLength);
	char* deviceDataPtr;
	cudaMalloc((void**)&deviceDataPtr, dataBlobLength);
	char* deviceVoidsPtr;
	cudaMalloc((void**)&deviceVoidsPtr, voidsBlobLength);

	char* deviceRFlagsPtr;
	cudaMalloc((void**)&deviceRFlagsPtr, lineMaskLenght);
	char* deviceGFlagsPtr;
	cudaMalloc((void**)&deviceGFlagsPtr, lineMaskLenght);
	char* deviceBFlagsPtr;
	cudaMalloc((void**)&deviceBFlagsPtr, lineMaskLenght);
	char* deviceAFlagsPtr;
	cudaMalloc((void**)&deviceAFlagsPtr, lineMaskLenght);

	cudaMemcpy(deviceSizingsPtr, sizingsBlob, sizingsBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceRegistryPtr, registryBlob, registryBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataPtr, dataBlob, dataBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVoidsPtr, voidsBlob, voidsBlobLength, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceRFlagsPtr, rFlagsBlob, lineMaskLenght, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGFlagsPtr, gFlagsBlob, lineMaskLenght, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBFlagsPtr, bFlagsBlob, lineMaskLenght, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceAFlagsPtr, aFlagsBlob, lineMaskLenght, cudaMemcpyHostToDevice);


	int* offsets = (int*)deviceRegistryPtr;
	int offsetsLength = spritesCount * 4;
	short* widths = (short*)(deviceRegistryPtr + offsetsLength);
	int widthsLength = spritesCount * 2;
	short* heights = (short*)(deviceRegistryPtr + offsetsLength + widthsLength);
	int heightsLength = widthsLength;

	short* sizingWidths = (short*)deviceSizingsPtr;
	int sizingWidthsLength = sizingsCount * 2;
	short* sizingHeights = (short*)(deviceSizingsPtr + sizingWidthsLength);
	int sizingHeightsLength = sizingWidthsLength;

	unsigned char* r = (unsigned char*)deviceDataPtr;
	unsigned char* g = (unsigned char*)(deviceDataPtr + dataBlobLineLength);
	unsigned char* b = (unsigned char*)(deviceDataPtr + dataBlobLineLength * 2);
	unsigned char* a = (unsigned char*)(deviceDataPtr + dataBlobLineLength * 3);

	for (size_t i = 0; i < 16; i++)
		printf("R flag %d: %d\n", i, (unsigned char)rFlagsBlob[i]);
	for (size_t i = 0; i < 16; i++)
		printf("G flag %d: %d\n", i, (unsigned char)gFlagsBlob[i]);
	for (size_t i = 0; i < 16; i++)
		printf("B flag %d: %d\n", i, (unsigned char)bFlagsBlob[i]);
	for (size_t i = 0; i < 16; i++)
		printf("A flag %d: %d\n", i, (unsigned char)aFlagsBlob[i]);

	dim3 block(BLOCK_SIZE);
	dim3 grid(spritesCount, spritesCount, sizingsCount); //Сайзингов будет меньше, чем спрайтов, так что сайзинги записываем в z
	mainKernel << <grid, block, BLOCK_SIZE * contextBits / 32 + 1 >> > (sizingsCount, sizingWidths, sizingHeights, spritesCount, offsets, widths, heights, r, g, b, a);

	cudaDeviceSynchronize();

	cudaFree(deviceSizingsPtr);
	cudaFree(deviceRegistryPtr);
	cudaFree(deviceDataPtr);
	cudaFree(deviceVoidsPtr);

	cudaFree(deviceRFlagsPtr);
	cudaFree(deviceGFlagsPtr);
	cudaFree(deviceBFlagsPtr);
	cudaFree(deviceAFlagsPtr);

	free(blob);

	return 0;
}
