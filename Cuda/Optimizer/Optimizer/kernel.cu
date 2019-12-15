﻿
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
48 кб на блок максимально эффективно, в том числе для уменьшения дайверженси и увеличения оккупации своими силами. А это как раз, что достигается тем, о чем
я рассуждал в позапрошлом абзаце. Но тогда я не знал про это ограничение. Теперь нам остается только решить на что потратить эти 48 кб.

Предположим, что средний размер спрайта у нас будет 256х256.
Тогда одна битовая маска у нас будет занимать 8кб. Мы можем взять, например:
1 карту пустоты полную.
4 битовых карты для 4х каналов.
Оставшиеся 8 кб - это 8 байтов информации на поток. Пока что из них я вижу 6 уйдет на контекст - 2 шорта для описания своей области и кандидатской и по 1
байту на описание текущей точки остановки. Вообще, я могу выделить и по полтора байта, таким образом сразу предупредив поддержку спрайтов до 4096х4096. И
таким образом у меня останется всего 1 неиспользованный байт.


8.12.2019
Ок, хранить по 1.5 байта на контекст, пожалуй, не буду, потому что я не уверен, как это будет записываться, думаю поведение при записи может быть undefined.
Так что выделю сразу 2 байта.

А вообще да, мы конечно молодцы, выделили 2 байта, типа поддерживаем спрайты до 65536х65536, а на деле все наши рассчеты были для спрайта размером 256х256.
Вообще с таким подходом максимум, что может потянуть 1 см - 512х256, тогда 1 блок займет всю шаред память под завязку. Поэтому мы должны иметь это в виду. В
будущем, возможно, надо будет сделать варианты кернелов для больших текстур - с некоторыми отключенными флагами например. В принципе, если оставлять размер
контекста, каким мы его видим сейчас, 8 кб на блок, то максимальный поддерживаемый размер спрайта, который может использовать хотя бы 1 вспомогательную мапу
(саму полезную, пускай будет войдмапу) - это 1024х704 - (96Кб - 8Кб контекста) * 1024 * 8 = 720_896 бит информации доступной для записи какого-нибудь флага,
т.е. меньше 1мегапикселя, довольно скромно. Т.е. это граничный вариант, при котором возможно кеширование хоть чего-нибудь, дальнейшее увеличение размера
приведет к отсутствию хоть какого-то кеширования и существенному замедлению производительности. Хотя можно все еще кешировать, просто неполную информацию -
часть информации будет с кешем, часть без кеша, тоже вариант, хоть какая-то доля кеширования все же будет.

Так ок, еще одна засада, мы сейчас раздали шорты областям, а ведь если они будут шортами, то они смогут обозначить себя лишь на области 256х256. Так стоп. А
нафига мне хранить области и еще дополнительно координаты? Ведь это одно и то же! В общем, мне нужно всего лишь по 4 байта для своей и кандидатской области.
При этом обе можно хранить в виде координат. Хотя нет, попробую сначала хранить их порядковыми номерами, потом посмотрю, что удобнее.

Так, так, так, еще одна засада - я рассчитал все значения для одного спрайта. Для двух, потребуется в 2 раза больше памяти. Этого мы себе позволить не можем.
Вообще вариантов с использованием всех 96кб одним блоком не может быть, потому что половина см будет простаивать. Таким образом нам надо по-другому
распределить выделенные 48 кб.

8 кб на контекст.
16 кб на войдмапы
и остается всего 24 кб. Мы можем взять еще по одному флагу, скажем r-флаг.
И у нас останется неиспользованными еще 8 кб. Может быть есть смысл их оставить пока что неиспользованными. Т.к. эффективно использовать именно 8 кб я сейчас
не могу. А в будущем, возможно они понадобятся.

Ок, новая проблема. Динамически мы можем инициализировать лишь 1 значение. Следовательно надо будет сделать 1 общий массив, содержащий все флаги и кеши и
потом из него сделать нужные нам подмассивы. Структура этого общего массива должна быть такой: ourR + candidateR + ourVoid...

Придумал. Нам совсем не обязательно кашировать оба войд-массива. То, что мы - войд, мы можем посчитать сами довольно эффективно, для этого не нужен кеш, т.к.
считать придется всего 1 раз. И получается что у нас остается еще 16Кб, которые мы можем потратить на еще 1 флаг-канал.

Итак, структура общего массива: ourR + candidateR + ourG + candidateG + candidateVoid: 8 кб + 8 кб + 8 кб + 8 кб + 8 кб = 40 кб. Это мы пока что не
рассматриваем другие варианты кроме спрайтов 256х256. Далее можно будет не только сделать варианты для большего размера текстур с меньшими оптимизациями, но и
для меньшего размера текстур с большими оптимизациями.

Хм, кстати... Если я заранее знаю размеры структур, то зачем мне динамическая инициализация? Я могу просто проинициализировать массивы по-максимому для
данного типоразмера. Просто если спрайты окажутся меньше, этот массив окажется недозаполненным. Но я все равно не смогу эту недозаполненную память
использовать, потому что я тут должен ориентироваться на максимальный размер. Ок, делаем так тогда - это облегчит инициализацию.

Ок, записываем мы флаги в таком порядке. Для каждого спрайта записываем поочередно все пиксели. Для каждого потока нам надо будет определить сперва смещение
спрайта. В принципе мы можем использовать регистр. Просто делить значение на 4. Я проверил - нет нужды делать на 4, т.к. в регистре записан сдвиг по каналам,
т.е. по байтам.

Так, нет, стоп, нельзя просто так взять сдвиг из регистра, т.к. флаги мы записываем побитно, поэтому эти оффсеты не будут точны. Так что надо еще сделать
битовый регистр.

Ок, это самая муторная часть, но потом уже будет полегче и поинтереснее...

9.12.2019
Ок, кажись я кое-чего не учел. А именно - отступы сайзингов. С кешированием все понятно - мы кешируем все флаги данного спрайта без исключения. А вот с
войдмапами совсем другая история. Там войдмапы имеют другую размерность нежели спрайты, поэтому их будет меньше, но это не столь важно. Непонятно откуда брать
размер данных войдмапа каждого спрайта. Мы их тупо не присылаем. Надо прислать, опять в регистр, наверно. Так, нет, стоп. А откуда ж регистру знать об этом,
если он вообще не учитывает сайзинги? Блин, надо тогда самому вычислять что ли? Не, это нереально. Надо отдельную структуру под это?

10.12.2019
Блин, пипец, конечно, сколько разных структур данных, очень трудно это все сразу в голове держать.

Ок, нужно расписать раз и навсегда правильный вариант записи битовых оффсетов - битовые оффсеты всегда записываются по своим группам и с округлением до
байтов. У меня и войдмапы были неправильно записаны, и судя по всему еще и флаги щас надо будет переделывать.

Ок, похоже я еще не начал, а уже приближаюсь к максимально возможному числу регистров процессора на поток. На самом деле число регистров на см - всего 64к. В
рассчете на поток при наших плановых 2048 потоках это всего 31.25 потоков. Пипец какой-то. Т.е. это не только мегасложно удержать все в голове, но еще и надо
при этом стараться переиспорльзовать переменные, т.е. они скорее всего будут с малоговорящими названиями. Круто.

Ок, у меня уже только в качестве аргументов функции используются 18 регистров. И где-то на 30й-31й переменной оно отваливается. И это только то, что вижу я.
Там еще возможно при разворачивании выражений добавляется, скорее всего до максимальных 62х регистров на поток. Короче надо врубить режим максимальной
экономии.

11.12.2019
Давайте посчитаем как нам уложиться в 30 регистров.

Ну, во-первых, надо избавляться от такой раскоши, как хранение ссылок на однородные части массивов отдельно. r, g, b и a идут лесом, вместо них rgba.
sizingsCount и spritesCount непонятно что вообще делают в аргументах - им место в контантах.
В принципе вообще все, что не массивы может идти в константы. Хотя, собственно у нас таких только 2... ну ладно.
Хранение отдельных составляющих blockIdx, естественно, тоже не имеет смысла.
Отдельные каналы флагов тоже уходят.
На самом деле не обязательно сейчас прямо жестко оптимизировать. Достаточно, чтобы хотя бы все заработало и протестировать. А потом можно будет уже жестко оптимизировать.
Не вижу причин voidLengths не кинуть в константы. Они маленькие и запрашиваются потоками не параллельно. Ну хотя как немного? Для такого небольшого набора как 40 спрайтов и 22 сайзинга - это 4 * 40 * 22 = 3520 байт. Константной
памяти у нас всего 64 кб. Допустим, 4 кб мы займем единичными константами. Оставшихся 60Кб хватит на то, чтобы разместить оффсеты для 2792 спрайтов при 22х сайзингах. Ну, вполне реальная цифра для большого проекта. Для бОльших
циферь, думаю, можно будет что-нибудь придумать.
Сайзинги тоже идут в константы.
Думаю, весь регистр тоже туда идет. Это для условных 2000 спрайтов 24 кб


12.12.2019
Ок, оказалось, что нельзя делать константы из массивов, не зная их длину заранее. Для каких-то данных это не проблема, для других это решаемо с некоторыми оговорками, для третьих это нелья делать вообще.

А вот еще один прием. Нам ведь не нужны особо больше константы. Можно заблокировать кол-во сайзингов, скажем, на 22, и все оставшееся место распределить пропорционально этому кол-ву сайзигнов. Тогда и выяснится максимальное
кол-во спрайтов. Точнее даже не спрайтов а в целом пикселей.

Итак, у нас всего пока что занято 12 байтов единичными значениями, все остальное - массивы. Остально состоит из следующего:
22 * (sizeof(short) + 2) - все сайзинги. = 22*4 = 88
spritesCount * (sizeof(int) + sizeof(int) + sizeof(short) + sizeof(short)) - битовые и байтовые сдвиги спрайтов (int), а также их ширина и высота (short) =  12 * spritesCount
spritesCount * 22 * sizeof(int) - сдвиги карты пустот. = 88 * spritesCount
Итого, как ни странно уравнение для неизвестного получилось 100 * spritesCount = 64кб - 100б. Решив его мы получаем spritesCount = 654.36. Мда, маловато. Наверное придется исключать карты пустот. Тогда spritesCount получится
равным 5453. Уже более-менее. На самом деле можно сделать отдельные версии кернела для разного кол-ва спрайтов, с теми или иными ограничениями и трейдофами.
*/

#define BLOCK_SIZE 1024
#define DIVERGENCE_CONTROL_CYCLE 128 //Через какое кол-во операций мы проверяем на необходимость ужатия, чтобы избежать дивергенции?
#define DIVERGENCE_CONTROL_THRESHOLD 32 //Если какое число потоков простаивают надо ужимать? Думаю это значение вряд ли когда-нибудь изменится.
#define REGISTRY_STRUCTURE_LENGTH 12
#define SIZING_STRUCTURE_LENGTH 4
#define DATA_STRUCTURE_LENGTH 4
#define RESERVED_DATA_LENGHT 2

#define MAX_FLAGS_LENGTH_FOR_SPRITE 8192 //Для 256х256 спрайта он именно такой

__constant__ short SizingsCount;
__constant__ short SpritesCount;
__constant__ int ByteLineLength;
__constant__ int BitLineLength;

__constant__ short SizingWidths[22];
__constant__ short SizingHeights[22];
__constant__ int SpriteByteOffsets[654];
__constant__ int SpriteBitOffsets[654];
__constant__ short SpriteWidths[654];
__constant__ short SpriteHeights[654];
__constant__ int VoidOffsets[14388];

__global__ void mainKernel(unsigned char* rgbaData, unsigned char* voids, unsigned char* rgbaFlags)
{
	//int ourSpriteIndex = blockIdx.x;
	//int candidateSpriteIndex = blockIdx.y;
	//int sizingIndex = blockIdx.z;

	//int ourByteOffset = SpriteByteOffsets[blockIdx.x];
	//int ourBitOffset = SpriteBitOffsets[blockIdx.x];
	short ourWidth = SpriteWidths[blockIdx.x];
	short ourHeight = SpriteHeights[blockIdx.x];
	int ourSquare = ourWidth * ourHeight;
	int ourBitsSquare = ourSquare / 8;
	if (ourSquare % 8 != 0)
		ourBitsSquare++;

	/*int candidateByteOffset = SpriteByteOffsets[blockIdx.y];
	int candidateBitOffset = SpriteBitOffsets[blockIdx.y];*/
	int candidateWidth = SpriteWidths[blockIdx.y];
	int candidateHeight = SpriteHeights[blockIdx.y];
	int candidateSquare = candidateWidth * candidateHeight;
	int candidateBitsSquare = candidateSquare / 8;
	if (candidateSquare % 8 != 0)
		candidateBitsSquare++;

	short sizingWidth = SizingWidths[blockIdx.z];
	short sizingHeight = SizingHeights[blockIdx.z];
	/*if (threadIdx.x == 0)
		printf("Hello from block! My sprite is #%d (width %d, height %d) and I work with sprite %d (width %d, height %d) and sizing %d (width %d, height %d) \n", blockIdx.x, ourWidth, ourHeight, blockIdx.y, candidateWidth, candidateHeight, blockIdx.z, sizingWidth, sizingHeight);
	return;*/

	__shared__ int ourAreaContexts[BLOCK_SIZE];
	__shared__ int candidateAreaContexts[BLOCK_SIZE];

	int myArea = threadIdx.x;
	int candidateArea = threadIdx.x;

	ourAreaContexts[threadIdx.x] = myArea;
	candidateAreaContexts[threadIdx.x] = candidateArea;

	/*printf("Hello World!\n");
	return;*/
	//Так, ок, с инициализацией контекста вроде разобрались. Сейчас нам надо тупо скопировать всю нужную инфу в шаред-мемори

	/*
		Дальше мы просто загружаем кешированные флаги, для нашей и кандидатской текстур в полном объеме. Для 256х256 спрайтов, размер должен быть 8192 байт.
	*/

	__shared__ char ourRFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	__shared__ char candidateRFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	__shared__ char ourGFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	__shared__ char candidateGFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	__shared__ char candidateVoidMap[MAX_FLAGS_LENGTH_FOR_SPRITE];//Экономим регистры
	//__shared__ char cachedFlags[MAX_FLAGS_LENGTH_FOR_SPRITE * 5];

	int numberOfTimesWeNeedToLoadSelf = ourBitsSquare / BLOCK_SIZE;
	if (ourBitsSquare % BLOCK_SIZE != 0)
		numberOfTimesWeNeedToLoadSelf++;

	/*printf("ourSquare = %d, candidateSquare = %d\n", ourSquare, candidateSquare);
	return;*/

	int numberOfTimesWeNeedToLoadCandidate = candidateBitsSquare / BLOCK_SIZE;
	if (candidateBitsSquare % BLOCK_SIZE != 0)
		numberOfTimesWeNeedToLoadCandidate++;

	for (size_t i = 0; i < numberOfTimesWeNeedToLoadSelf; i++)
	{
		int byteAddress = i * BLOCK_SIZE + threadIdx.x;
		if (byteAddress >= ourBitsSquare)
			continue;
		ourRFlags[byteAddress] = rgbaFlags[SpriteBitOffsets[blockIdx.x] + byteAddress];
		ourGFlags[byteAddress] = rgbaFlags[BitLineLength + SpriteBitOffsets[blockIdx.x] + byteAddress];
	}

	//if (blockIdx.x == 7 && blockIdx.y == 7 && blockIdx.z == 18) //Так мы обойдемся без повторов, только 1 блок будет логировать
	//	printf("rgbaFlags[%d] = %d\n", threadIdx.x, rgbaFlags[threadIdx.x]);
	//printf("BitLineLength = %d, ByteLineLength = %d, SpritesCount = %d, SizingsCount = %d\n", BitLineLength, ByteLineLength, SpritesCount, SizingsCount);

	for (size_t i = 0; i < numberOfTimesWeNeedToLoadCandidate; i++)
	{
		int byteAddress = i * BLOCK_SIZE + threadIdx.x;
		if (byteAddress >= candidateBitsSquare)
			continue;
		candidateRFlags[byteAddress] = rgbaFlags[SpriteBitOffsets[blockIdx.y] + byteAddress];
		candidateGFlags[byteAddress] = rgbaFlags[BitLineLength + SpriteBitOffsets[blockIdx.y] + byteAddress];
	}

	int candidateWidthMinusSizing = candidateWidth - sizingWidth;
	int candidateHeightMinusSizing = candidateHeight - sizingHeight;
	int candidateVoidAreaSquare = candidateWidthMinusSizing * candidateHeightMinusSizing;
	int candidateVoidAreaBitSquare = candidateVoidAreaSquare / 8;
	if (candidateVoidAreaSquare % 8 != 0)
		candidateVoidAreaBitSquare++;

	int numberOfTimesWeNeedToLoadVoid = candidateVoidAreaBitSquare / BLOCK_SIZE;
	if (candidateVoidAreaBitSquare % BLOCK_SIZE != 0)
		numberOfTimesWeNeedToLoadVoid++;

	int candidateVoidMapOffset = VoidOffsets[blockIdx.y * SizingsCount + blockIdx.z];
	unsigned char* candidateVoidMapGlobal = voids + candidateVoidMapOffset;
	//if (blockIdx.x == 7 && blockIdx.y == 7 && blockIdx.z == 18 && threadIdx.x < candidateVoidAreaSquare)
	//{
	//	int candidateX = threadIdx.x / candidateHeightMinusSizing;
	//	int candidateY = threadIdx.x % candidateHeightMinusSizing;
	//	printf("	void (%d, %d): %d\n", candidateX, candidateY, candidateVoidMapGlobal[threadIdx.x / 8] >> threadIdx.x % 8 & 1);
	//} //Проверили правильность апрсинга войдмап


	for (size_t i = 0; i < numberOfTimesWeNeedToLoadVoid; i++)
	{
		int voidByteAddress = i * BLOCK_SIZE + threadIdx.x;
		if (voidByteAddress >= candidateVoidAreaBitSquare)
			continue;
		candidateVoidMap[voidByteAddress] = candidateVoidMapGlobal[voidByteAddress];
		//candidateVoidMap[voidByteAddress] = voids[SpriteBitOffsets[blockIdx.y]];
	}

	__syncthreads(); //Обязательна синхронизация для того, чтобы потоки которые не выполняли загрузку в шаред-память не начали с этой шаред памятью работать пока другие в нее еще не все загрузили.

	//Проверяем, что все скопировалось правильно. Для этого выбираем случайный спрайт и логируем его флаги. Пускай будет спрайт №7
	if (blockIdx.x == 7 && blockIdx.y == 7 && blockIdx.z == 18) //Так мы обойдемся без повторов, только 1 блок будет логировать
	{
		if (threadIdx.x < ourSquare)
		{
			int x = threadIdx.x / BLOCK_SIZE;
			int y = threadIdx.x % BLOCK_SIZE;
			printf("for pixel #%d (%d, %d) the flags of r and g are (%d, %d) == (%d, %d)\n", threadIdx.x, x, y, (ourRFlags[threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (ourGFlags[threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (candidateRFlags[threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (candidateGFlags[threadIdx.x / 8] >> (threadIdx.x % 8)) & 1);
		}
	} //Проверил, работает


	if (blockIdx.x == 7 && blockIdx.y == 7 && blockIdx.z == 18 && threadIdx.x < candidateVoidAreaSquare)
	{
		int candidateX = threadIdx.x / candidateHeightMinusSizing;
		int candidateY = threadIdx.x % candidateHeightMinusSizing;
		printf("	void (%d, %d): %d\n", candidateX, candidateY, candidateVoidMap[threadIdx.x / 8] >> threadIdx.x % 8 & 1);
	} //Проверили правильность апрсинга войдмап
}

//
//__global__ void testKernel(int sizingsCount, short* sizingWidths, short* sizingHeights, int spritesCount, int* byteOffsets, int* bitOffsets, short* widths, short* heights, unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a)
//{
//	if (threadIdx.x == 0)
//	{
//		printf("%d\n", threadIdx.x);
//		return;
//	}
//	int ourSpriteIndex = blockIdx.x;
//	int candidateSpriteIndex = blockIdx.y;
//	int sizingIndex = blockIdx.z;
//
//	int ourOffset = offsets[ourSpriteIndex];
//}

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

	//blob состоит из мета-инфы, которую мы не используем в рассчетах, и основной инфы. Мета - 4 байта длина и собственно данные.
	int metaLength = bit_converter::GetInt(blob, 0);
	int combinedDataOffset = metaLength + sizeof(int); //указатель на начало массива основных данных

	short spritesCount = bit_converter::GetShort(blob, combinedDataOffset + RESERVED_DATA_LENGHT); // первые 2 байта основных данных зарезервированы, вторые - кол-во спрайтов.
	cudaMemcpyToSymbol(SpritesCount, &spritesCount, sizeof(short)); //Сразу записываем их в константы устройства
	short sizingsCount = bit_converter::GetShort(blob, combinedDataOffset + RESERVED_DATA_LENGHT + sizeof(short)); //третьи 2 байта - кол-во сайзингов
	cudaMemcpyToSymbol(SizingsCount, &sizingsCount, sizeof(short)); //Тоже записываем сразу туда.

	//Определяем массивы и длины данных сайзингов и регистра

	//Сначала сайзинги...
	char* sizingsBlob = blob + combinedDataOffset + RESERVED_DATA_LENGHT + sizeof(short) * 2;
	int sizingsBlobLenght = sizingsCount * SIZING_STRUCTURE_LENGTH; //Сайзинги состоят из 2 шортов - х и у
	//Записываем сайзинги на девайс. Они там идут последовательно, сначала иксы потом игрики
	int sizingsLineLength = sizeof(short) * sizingsCount;
	cudaMemcpyToSymbol(SizingWidths, sizingsBlob, sizingsLineLength);
	cudaMemcpyToSymbol(SizingHeights, sizingsBlob + sizingsLineLength, sizingsLineLength);


	char* registryBlob = sizingsBlob + sizingsBlobLenght;
	int registryBlobLength = spritesCount * REGISTRY_STRUCTURE_LENGTH; //регистр на данный момент состоит из 2 шортов и 2 интов, длина структуры задается через REGISTRY_STRUCTURE_LENGTH
	//Записываем регистр на девайс. Они там идут последовательно, сначала байтовые оффсеты потом битовые, потом иксы, потом игрики
	int registryLineCount = spritesCount * sizingsCount;
	cudaMemcpyToSymbol(SpriteByteOffsets, registryBlob, spritesCount * sizeof(int));
	cudaMemcpyToSymbol(SpriteBitOffsets, registryBlob + spritesCount * sizeof(int), spritesCount * sizeof(int));
	cudaMemcpyToSymbol(SpriteWidths, registryBlob + spritesCount * sizeof(int) * 2, spritesCount * sizeof(short));
	cudaMemcpyToSymbol(SpriteHeights, registryBlob + spritesCount * (sizeof(int) * 2 + sizeof(short)), spritesCount * sizeof(short));

	//Дальше идет длина 1 канала цвета
	int byteLineLength = bit_converter::GetInt(registryBlob + registryBlobLength, 0);
	cudaMemcpyToSymbol(ByteLineLength, &byteLineLength, sizeof(int)); //Сразу записываем ее в константы

	//Дальше идут данные. А зная длину 1 канала цвета, мы можем легко посчитать общую длину массива цветов
	char* rgbaBlob = registryBlob + registryBlobLength + sizeof(int);
	int rgbaBlobLength = byteLineLength * DATA_STRUCTURE_LENGTH; //Длина структуры основных данных у нас 4 - по 1 байту на канал.
	//Сразу записываем их в глобальную память
	char* deviceRgbaDataPtr;
	cudaMalloc((void**)&deviceRgbaDataPtr, rgbaBlobLength);
	cudaMemcpy(deviceRgbaDataPtr, rgbaBlob, rgbaBlobLength, cudaMemcpyHostToDevice);

	/*
		Это были основные данные. Дальше идут вспомогательные.
	*/


	//Следующими за данными о цвете идут оффсеты карт пустот
	int voidMapsCount = spritesCount * sizingsCount; //Карта пустот есть отдельно для каждого спрайта каждого сайзинга
	int* voidMapsOffsets = (int*)(rgbaBlob + rgbaBlobLength);
	int voidMapsOffsetsLength = voidMapsCount * sizeof(int);
	cudaMemcpyToSymbol(VoidOffsets, voidMapsOffsets, voidMapsOffsetsLength); //Пишем ее в константы

	//Потом длина уже непосредственно карт и за ней - сами данные карты
	int voidsBlobLength = bit_converter::GetInt((char*)(voidMapsOffsets + voidMapsCount), 0);
	char* voidsBlob = (char*)(voidMapsOffsets + voidMapsCount) + sizeof(int);
	//Пишем эти данные в глобал мемори
	char* deviceVoidsPtr;
	cudaMalloc((void**)&deviceVoidsPtr, voidsBlobLength);
	cudaMemcpy(deviceVoidsPtr, voidsBlob, voidsBlobLength, cudaMemcpyHostToDevice);

	//Дальше длина канала основных данных в битах
	int bitLineLength = bit_converter::GetInt(voidsBlob + voidsBlobLength, 0);
	cudaMemcpyToSymbol(BitLineLength, &bitLineLength, sizeof(int)); // Сразу записываем ее
	char* rgbaFlags = voidsBlob + voidsBlobLength + sizeof(int);
	int rgbaFlagsLength = bitLineLength * DATA_STRUCTURE_LENGTH;
	//Пишем их тоже в глобальную
	char* deviceRgbaFlagsPtr;
	cudaMalloc((void**)&deviceRgbaFlagsPtr, rgbaFlagsLength);
	cudaMemcpy(deviceRgbaFlagsPtr, rgbaFlags, rgbaFlagsLength, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE);
	dim3 grid(spritesCount, spritesCount, sizingsCount); //Сайзингов будет меньше, чем спрайтов, так что сайзинги записываем в z
	mainKernel << <grid, block >> > ((unsigned char*)deviceRgbaDataPtr, (unsigned char*)deviceVoidsPtr, (unsigned char*)deviceRgbaFlagsPtr);

	cudaDeviceSynchronize();

	cudaFree(deviceRgbaDataPtr);
	cudaFree(deviceVoidsPtr);
	cudaFree(deviceRgbaFlagsPtr);

	free(blob);

	return 0;
}
