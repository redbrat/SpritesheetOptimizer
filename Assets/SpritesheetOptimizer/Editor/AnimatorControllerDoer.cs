using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using static SpritesheetOptimizerWindow;

public static class AnimatorControllerDoer
{
    private static UnityOptimizedSpritesStructure[] _chunksInfos;

    internal static void Do(AnimatorController originalCtrlr, string folderPath, params UnityOptimizedSpritesStructure[] chunksInfo)
    {
        _chunksInfos = chunksInfo;

        var optName = $"{originalCtrlr.name}-optimized";
        var path = Path.Combine(folderPath, $"{optName}.controller");
        Debug.Log($"Do {path}");
        var optCtrlr = AnimatorController.CreateAnimatorControllerAtPath(path);
        optCtrlr.RemoveLayer(0);

        var optGo = new GameObject(optName);

        var animationClipsFolder = Path.Combine(folderPath, "AnimationClips");
        if (!Directory.Exists(animationClipsFolder))
            Directory.CreateDirectory(animationClipsFolder);

        var originalToOptObjectReferences = new Dictionary<UnityEngine.Object, UnityEngine.Object>();

        for (int i = 0; i < originalCtrlr.layers.Length; i++)
        {
            var optLayer = getOptimizedLayer(originalCtrlr.layers[i], optGo, originalToOptObjectReferences, animationClipsFolder);
            optCtrlr.AddLayer(optLayer);
        }

        var prefabPath = Path.Combine(folderPath, $"{optName}.prefab");
        PrefabUtility.SaveAsPrefabAsset(optGo, prefabPath);
        UnityEngine.Object.DestroyImmediate(optGo);
    }

    private static T getOptReference<T>(T original, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder) where T : UnityEngine.Object
    {
        if (original == null)
            return null;
        if (!originalToOptObjectReferences.ContainsKey(original))
        {
            if (typeof(T).Equals(typeof(AnimatorStateMachine)))
                return (T)(UnityEngine.Object)getOptimizedStateMachine(original as AnimatorStateMachine, prefab, originalToOptObjectReferences, animationClipsFolder);
            else if (typeof(T).Equals(typeof(AnimatorState)))
                return (T)(UnityEngine.Object)getOptimizedState(original as AnimatorState, prefab, originalToOptObjectReferences, animationClipsFolder);
            else if (typeof(Motion).IsAssignableFrom(typeof(T)))
                return (T)(UnityEngine.Object)getOptimizedMotion(original as Motion, prefab, originalToOptObjectReferences, animationClipsFolder);
            else
                throw new ApplicationException($"Unknown reference type occured :{typeof(T).FullName}!");
        }
        else
            return (T)originalToOptObjectReferences[original];
    }

    private static AnimatorControllerLayer getOptimizedLayer(AnimatorControllerLayer originalLayer, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optLayer = new AnimatorControllerLayer();

        optLayer.blendingMode = originalLayer.blendingMode;
        optLayer.defaultWeight = originalLayer.defaultWeight;
        optLayer.iKPass = originalLayer.iKPass;
        optLayer.name = originalLayer.name;
        optLayer.stateMachine = getOptReference(originalLayer.stateMachine, prefab, originalToOptObjectReferences, animationClipsFolder);
        optLayer.syncedLayerAffectsTiming = originalLayer.syncedLayerAffectsTiming;
        optLayer.syncedLayerIndex = originalLayer.syncedLayerIndex;

        return optLayer;
    }

    private static AnimatorStateMachine getOptimizedStateMachine(AnimatorStateMachine originalStateMachine, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optStateMachine = new AnimatorStateMachine();
        originalToOptObjectReferences.Add(originalStateMachine, optStateMachine);

        optStateMachine.anyStatePosition = originalStateMachine.anyStatePosition;
        optStateMachine.entryPosition = originalStateMachine.entryPosition;
        optStateMachine.exitPosition = originalStateMachine.exitPosition;
        optStateMachine.name = originalStateMachine.name;
        optStateMachine.parentStateMachinePosition = originalStateMachine.parentStateMachinePosition;

        for (int i = 0; i < originalStateMachine.states.Length; i++)
            optStateMachine.AddState(getOptReference(originalStateMachine.states[i].state, prefab, originalToOptObjectReferences, animationClipsFolder), originalStateMachine.states[i].position);

        //for (int i = 0; i < originalStateMachine.anyStateTransitions.Length; i++)
        //{
        //    originalStateMachine.AddAnyStateTransition
        //}

        return optStateMachine;
    }

    private static AnimatorState getOptimizedState(AnimatorState originalAnimatorState, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optAnimatorState = new AnimatorState();
        originalToOptObjectReferences.Add(originalAnimatorState, optAnimatorState);

        optAnimatorState.cycleOffset = originalAnimatorState.cycleOffset;
        optAnimatorState.cycleOffsetParameter = originalAnimatorState.cycleOffsetParameter;
        optAnimatorState.cycleOffsetParameterActive = originalAnimatorState.cycleOffsetParameterActive;
        optAnimatorState.iKOnFeet = originalAnimatorState.iKOnFeet;
        optAnimatorState.mirror = originalAnimatorState.mirror;
        optAnimatorState.mirrorParameter = originalAnimatorState.mirrorParameter;
        optAnimatorState.mirrorParameterActive = originalAnimatorState.mirrorParameterActive;
        //optAnimatorState.motion = getOptimizedMotion(originalAnimatorState.motion, structure, originalToOptObjectReferences, animationClipsFolder);
        optAnimatorState.motion = getOptReference(originalAnimatorState.motion, prefab, originalToOptObjectReferences, animationClipsFolder);
        optAnimatorState.name = originalAnimatorState.name;
        optAnimatorState.speed = originalAnimatorState.speed;
        optAnimatorState.speedParameter = originalAnimatorState.speedParameter;
        optAnimatorState.speedParameterActive = originalAnimatorState.speedParameterActive;
        optAnimatorState.tag = originalAnimatorState.tag;
        optAnimatorState.timeParameter = originalAnimatorState.timeParameter;
        optAnimatorState.timeParameterActive = originalAnimatorState.timeParameterActive;
        optAnimatorState.writeDefaultValues = originalAnimatorState.writeDefaultValues;

        optAnimatorState.transitions = getOptimizedAnimatorStateTransition(originalAnimatorState.transitions, prefab, originalToOptObjectReferences, animationClipsFolder);

        return optAnimatorState;
    }

    private static Motion getOptimizedMotion(Motion originalMotion, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        if (!(originalMotion is AnimationClip))
            throw new ApplicationException($"Unknown type of motion - {originalMotion.GetType().FullName}. Never done this before...");

        var originalAnimationClip = originalMotion as AnimationClip;
        var optMotion = new AnimationClip();
        originalToOptObjectReferences.Add(originalMotion, optMotion);

        optMotion.frameRate = originalAnimationClip.frameRate;
        AnimationUtility.SetAnimationEvents(optMotion, getOptimizedAnimationClipEvents(originalAnimationClip.events));
        optMotion.hideFlags = originalAnimationClip.hideFlags;
        optMotion.legacy = originalAnimationClip.legacy;
        optMotion.localBounds = originalAnimationClip.localBounds;
        optMotion.name = originalAnimationClip.name;
        optMotion.wrapMode = originalAnimationClip.wrapMode;

        /*
         * Собственно, вся работа происходит здесь. Перебираем дорожки анимации, имеем доступ к используемым спрайтам.
         * Наша задача - повторить спрайт в оптимизированном виде на данном конкретном временном отрезке на любом количестве дорожек.
         * Каждая дорожка это отдеальный спрайт, отображающий наш выбранного размера чанк оптимизированного спрайтшита. При этом гораздо большего
         * сжатия мы добъемся, если будем переиспользовать спрайты, т.е. если будет возможность менять из взаиморасположение. Таким образом для каждого 
         * спрайта будут 2 дорожки - собственно спрайт и localPosition его трансформации.
         */

        var objectReferenceBindings = AnimationUtility.GetObjectReferenceCurveBindings(originalAnimationClip);
        var bindings = AnimationUtility.GetCurveBindings(originalAnimationClip);
        if (bindings != default)
            for (int i = 0; i < bindings.Length; i++)
            {
                var originalBinging = bindings[i];
                var optBinding = originalBinging;//.MemberwiseClone();
                createWhatToBindTo(optBinding, prefab);
                var originalCurve = AnimationUtility.GetEditorCurve(originalMotion as AnimationClip, bindings[i]);
                //var optCurve = originalCurve.MemberwiseClone();
                setOptBindings(optMotion, originalCurve, optBinding);
                //AnimationUtility.SetEditorCurve(optMotion, optBinding, optCurve);
            }
        //var optimizedBindingsList = new List<EditorCurveBinding>();
        if (objectReferenceBindings != default)
            for (int i = 0; i < objectReferenceBindings.Length; i++)
            {
                var originalBinding = objectReferenceBindings[i];
                Debug.Log($"originalBinding.propertyName = {originalBinding.propertyName}");
                Debug.Log($"originalBinding.path = {originalBinding.path}");
                Debug.Log($"originalBinding.type = {originalBinding.type.FullName}");
                var keyframes = AnimationUtility.GetObjectReferenceCurve(originalAnimationClip, originalBinding);
                //createWhatToBindTo(originalBinding, prefab);
                setOptObjectReferenceBindings(optMotion, prefab, originalBinding, keyframes);
                //setOptimizedCurvesBasedOnOriginalCurve(optMotion, structure, originalBinding, keyframes);
                //AnimationUtility.SetObjectReferenceCurve(optMotion, originalBinding, optimizedBindingsList.ToArray());
            }

        var animationClipPath = Path.Combine(animationClipsFolder, $"{originalAnimationClip.name}.anim");
        AssetDatabase.CreateAsset(optMotion, animationClipPath);
        //AssetDatabase.SaveAssets();
        return optMotion;
    }

    private static GameObject createWhatToBindTo(EditorCurveBinding optBinding, GameObject prefab)
    {
        var path = optBinding.path;
        var targetGo = prefab;
        var address = new List<string>();
        if (!string.IsNullOrEmpty(path))
        {
            if (path.Contains('/'))
                address.AddRange(path.Split('/'));
            else
                address.Add(path);
        }
        return createWhatToBindToRecursively(prefab, address, optBinding);
    }

    private static GameObject createWhatToBindToRecursively(GameObject currentGo, List<string> address, EditorCurveBinding optBinding)
    {
        if (address.Count > 0)
        {
            var currentAddressee = address[0];
            address.RemoveAt(0);
            var addresseeTr = currentGo.transform.Find(currentAddressee);
            if (addresseeTr == default)
            {
                var newGo = new GameObject(currentAddressee);
                addresseeTr = newGo.transform;
                addresseeTr.SetParent(currentGo.transform);
            }
            return createWhatToBindToRecursively(addresseeTr.gameObject, address, optBinding); ;
        }

        var component = currentGo.GetComponent(optBinding.type);
        if (component == default)
            component = currentGo.AddComponent(optBinding.type);
        return currentGo;
    }

    private static void setOptBindings(AnimationClip optMotion, AnimationCurve originalCurve, EditorCurveBinding optBinding)
    {
        var optCurve = new AnimationCurve();
        var optKeyframes = new Keyframe[originalCurve.keys.Length];
        for (int i = 0; i < optKeyframes.Length; i++)
        {
            optKeyframes[i] = originalCurve.keys[i];
        }
        optCurve.keys = optKeyframes;
        AnimationUtility.SetEditorCurve(optMotion, optBinding, optCurve);
    }

    private class BindingTracks
    {
        public EditorCurveBinding SpriteBinding;
        public EditorCurveBinding TransformBindingX;
        public EditorCurveBinding TransformBindingY;
        public List<ObjectReferenceKeyframe?> SpriteKeyframes;
        public AnimationCurve TransformCurveX;
        public AnimationCurve TransformCurveY;
        public List<Keyframe?> TransformCurveKeyframesX; //Все эти приблуды нужны для того, чтобы не было проблем с работой с массивами структур
        public List<Keyframe?> TransformCurveKeyframesY;
    }

    private static void setOptObjectReferenceBindings(AnimationClip optMotion, GameObject prefab, EditorCurveBinding originalBinding, ObjectReferenceKeyframe[] keyframes)
    {
        /*
         * Ок, тут мы имеем на входе 1 дорожку - originalBinding и кифреймы, описанные keyframes. И задача у нас - 
         * проставить optMotion набор дорожек, заменяющих входящую дорожку.
         */

        /*
         * Ок, тут мы имеем на входе AnimationClip, в котором могут анимироваться какое угодно кол-во параметров - ссылочные и не очень.
         * Ссылочные могут быть как спрайтами так и не очень. Спрайты могут быть те, которые мы оптимизировали, а могут быть вообще левые
         * спрайты, которые надо просто скопировать. Поэтому мы должны иметь в общем две структуры: та, в которой оптимизируемые спрайты,
         * и другая, в которой все остальное. При этом может быть так что какая-то из этих структур не будет представлена вовсе.
         */

        /*
         * Ок, нет, на самом деле, мы тут на вход имеем 1 дорожку. В ней может быть что угодно. А результат мы должны вознать в ANimationClip
         */


        //Сначала нам надо понять сколько дорожек нам понадобится.
        var maxOptSpritesCount = 0;
        for (int i = 0; i < keyframes.Length; i++)
        {
            var keyframe = keyframes[i];
            if (keyframe.value == null)
                continue;

            if (!(keyframe.value is Sprite))
            {
                //Если мы анимируем не спрайт, то нас это не интересует - мы просто копируем ссылку
                if (maxOptSpritesCount < 1)
                    maxOptSpritesCount = 1;
                continue;
            }

            var chunksInfo = _chunksInfos.Where(info => info.Sprites.Contains(keyframe.value));
            if (!chunksInfo.Any()) //Если оптимизированной структуры не найдено, то просто ставим оригинал, т.е. 1 спрайт.
            {
                if (maxOptSpritesCount < 1)
                    maxOptSpritesCount = 1;
                continue;
            }

            var chunkedSprites = chunksInfo.First().Sprites;
            var chunks = default(SpriteChunkArrayWrapper);
            for (int j = 0; j < chunkedSprites.Length; j++)
            {
                if (chunkedSprites[j] == keyframe.value)
                {
                    chunks = chunksInfo.First().Chunks[j];
                    break;
                }
            }

            if (chunks.Array.Length > maxOptSpritesCount)
                maxOptSpritesCount = chunks.Array.Length;
        }
        Debug.Log($"Ок, максимальное кол-во дорожек у нас {maxOptSpritesCount}");

        //Создаем дорожки
        var optBindingsTracks = new List<BindingTracks>();
        for (int i = 0; i < maxOptSpritesCount; i++)
        {
            var binds = new BindingTracks();
            var spriteBinding = new EditorCurveBinding();
            var transformBindingX = new EditorCurveBinding();
            var transformBindingY = new EditorCurveBinding();

            var gameObjectName = $"OptimizerSpriteRenderer_{i}";
            spriteBinding.propertyName = originalBinding.propertyName;
            var path = gameObjectName;
            if (!string.IsNullOrEmpty(originalBinding.path))
                path = $"{originalBinding.path}/{path}";
            spriteBinding.path = path;
            spriteBinding.type = originalBinding.type;
            createWhatToBindTo(spriteBinding, prefab);

            binds.SpriteBinding = spriteBinding;

            binds.SpriteKeyframes = new  List<ObjectReferenceKeyframe?>();
            for (int j = 0; j < keyframes.Length; j++)
                binds.SpriteKeyframes.Add(new ObjectReferenceKeyframe());

            //Кол-во дорожек может быть больше 1 только в случае, если мы оптимизируем спрайт, во всех остальных случаях
            //мы оставляем оригинальные дорожки и ссылки. Т.ч. если дорожка 1, нам не нужны дополнительные дороги трансформации
            if (maxOptSpritesCount > 1)
            {
                transformBindingX.propertyName = "m_LocalPosition.x";
                path = gameObjectName;
                if (!string.IsNullOrEmpty(originalBinding.path))
                    path = $"{originalBinding.path}/{path}";
                transformBindingX.path = path;
                transformBindingX.type = typeof(Transform);
                createWhatToBindTo(transformBindingX, prefab);

                transformBindingY.propertyName = "m_LocalPosition.y";
                path = gameObjectName;
                if (!string.IsNullOrEmpty(originalBinding.path))
                    path = $"{originalBinding.path}/{path}";
                transformBindingY.path = path;
                transformBindingY.type = typeof(Transform);
                createWhatToBindTo(transformBindingY, prefab);

                binds.TransformBindingX = transformBindingX;
                binds.TransformBindingY = transformBindingY;

                binds.TransformCurveX = new AnimationCurve();
                binds.TransformCurveY = new AnimationCurve();
                binds.TransformCurveKeyframesX = new List<Keyframe?>();
                binds.TransformCurveKeyframesY = new List<Keyframe?>();

                for (int j = 0; j < keyframes.Length; j++)
                {
                    binds.TransformCurveKeyframesX.Add(new Keyframe());
                    binds.TransformCurveKeyframesY.Add(new Keyframe());
                }
            }

            optBindingsTracks.Add(binds);
        }

        var isOptimized = false;
        //Теперь мы проходимся по каждому кифрейму оригинальной дороги и инициализируем все альтернативные соответствующие кифреймы
        for (int i = 0; i < keyframes.Length; i++)
        {
            var keyframe = keyframes[i];
            if (keyframe.value == default)
            {
                for (int j = 0; j < optBindingsTracks.Count; j++)
                {
                    var key = optBindingsTracks[j].SpriteKeyframes[i].Value;
                    key.time = keyframe.time;
                    optBindingsTracks[j].SpriteKeyframes[i] = key;
                }

                continue;
            }
            if (!(keyframe.value is Sprite)) //Если у нас не спрайт, то дорожка всего 1
            {
                var key = optBindingsTracks[0].SpriteKeyframes[i].Value;
                key.time = keyframe.time;
                key.value = keyframe.value;
                optBindingsTracks[0].SpriteKeyframes[i] = key;
                continue;
            }

            var chunksInfo = _chunksInfos.Where(info => info.Sprites.Contains(keyframe.value));
            if (!chunksInfo.Any()) //Если оптимизированной структуры для спрайта на данном кадре не найдено, то просто ставим оригинал
            {
                {
                    var key = optBindingsTracks[0].SpriteKeyframes[i].Value;
                    key.time = keyframe.time;
                    key.value = keyframe.value;
                    optBindingsTracks[0].SpriteKeyframes[i] = key;
                }


                if (optBindingsTracks[0].TransformCurveX != default) //Такое может быть, если по какой-то причине у нас всего 1 дорожка
                {
                    {
                        var key = optBindingsTracks[0].TransformCurveKeyframesX[i].Value;
                        key.time = keyframe.time;
                        key.value = 0;
                        optBindingsTracks[0].TransformCurveKeyframesX[i] = key;
                    }

                    {
                        var key = optBindingsTracks[0].TransformCurveKeyframesY[i].Value;
                        key.time = keyframe.time;
                        key.value = 0;
                        optBindingsTracks[0].TransformCurveKeyframesY[i] = key;
                    }
                }

                for (int j = 1; j < optBindingsTracks.Count; j++)
                {
                    var key = optBindingsTracks[j].SpriteKeyframes[i].Value;
                    key.time = keyframe.time;
                    key.value = default;
                    optBindingsTracks[j].SpriteKeyframes[i] = key;
                }
                continue;
            }

            isOptimized = true;
            var chunkedSprites = chunksInfo.First().Sprites;
            var chunks = default(SpriteChunk[]);
            for (int j = 0; j < chunkedSprites.Length; j++)
            {
                if (chunkedSprites[j] == keyframe.value)
                {
                    chunks = chunksInfo.First().Chunks[j].Array;
                    break;
                }
            }

            for (int j = 0; j < optBindingsTracks.Count; j++)
            {
                if (j < chunks.Length)
                {
                    {
                        var key = optBindingsTracks[j].SpriteKeyframes[i].Value;
                        key.time = keyframe.time;
                        key.value = chunks[j].ChunkSprite;
                        optBindingsTracks[j].SpriteKeyframes[i] = key;
                    }

                    if (optBindingsTracks[j].TransformBindingX != default)
                    {
                        var keyX = new Keyframe();
                        keyX.time = keyframe.time;
                        keyX.value = chunks[j].Area.X / 100f;

                        var keyY = new Keyframe();
                        keyY.time = keyframe.time;
                        keyY.value = chunks[j].Area.Y / 100f;

                        optBindingsTracks[j].TransformCurveKeyframesX[i] = keyX;
                        optBindingsTracks[j].TransformCurveKeyframesY[i] = keyY;
                        //optBindingsTracks[j].TransformCurveX.keys[i] = keyX;
                        //optBindingsTracks[j].TransformCurveY.keys[i] = keyY;
                    }
                }
                else
                {
                    var key = optBindingsTracks[j].SpriteKeyframes[i].Value;
                    key.time = keyframe.time;
                    key.value = default;
                    optBindingsTracks[j].SpriteKeyframes[i] = key;
                }
            }
        }

        for (int i = 0; i < optBindingsTracks.Count; i++)
        {
            optBindingsTracks[i].TransformCurveX.keys = optBindingsTracks[i].TransformCurveKeyframesX.Select(v => v.Value).ToArray();
            optBindingsTracks[i].TransformCurveY.keys = optBindingsTracks[i].TransformCurveKeyframesY.Select(v => v.Value).ToArray();
        }

        if (isOptimized)
        {
            var counts = new Dictionary<int, int>();
            for (int i = 0; i < optBindingsTracks.Count; i++)
            {
                var keys = optBindingsTracks[i].SpriteKeyframes.Select(v => v.Value).ToArray();
                for (int j = 0; j < keys.Length; j++)
                {
                    if (!counts.ContainsKey(j))
                        counts.Add(j, 0);

                    var spriteChunk = keys[j].value;
                    if (spriteChunk != default)
                    {
                        var texture = (spriteChunk as Sprite).texture;
                        for (int x = 0; x < texture.width; x++)
                        {
                            for (int y = 0; y < texture.height; y++)
                            {
                                var pixel = texture.GetPixel(x, y);
                                if (pixel.a > 0f)
                                    counts[j]++;
                            }
                        }
                    }
                }
            }

            foreach (var count in counts)
            {
                Debug.Log($"count #{count.Key}: {count.Value}");
            }
        }

        for (int i = 0; i < optBindingsTracks.Count; i++)
        {
            var currentTracks = optBindingsTracks[i];
            AnimationUtility.SetObjectReferenceCurve(optMotion, currentTracks.SpriteBinding, currentTracks.SpriteKeyframes.Select(v => v.Value).ToArray());
            if (currentTracks.TransformCurveX != default)
            {
                AnimationUtility.SetEditorCurve(optMotion, currentTracks.TransformBindingX, currentTracks.TransformCurveX);
                AnimationUtility.SetEditorCurve(optMotion, currentTracks.TransformBindingY, currentTracks.TransformCurveY);
            }
        }

        //AnimationUtility.SetObjectReferenceCurve(optMotion, originalBinding, optimizedBindingsList.ToArray());
    }

    //private static void setOptimizedCurvesBasedOnOriginalCurve(AnimationClip optMotion, OptimizedControllerStructure structure, EditorCurveBinding originalBinding, ObjectReferenceKeyframe[] originalKeyframes, int resolutionX = 4, int resolutionY = 4)
    //{
    //    /*
    //     *      originalBinding содержит инфу о цели анимации. path - путь трансформации вида SecondSprite/Thrid, 
    //     * type - тип компонента, propertyName - имя сериализуемого поля. 
    //     *      originalKeyframes - просто содержит ссылки на объекты и время.
    //     *      optMotion - то, куда надо добавить оптимизированные кривые.
    //     * 
    //     * Тут нам надо проставить кривые новому оптимизированному AnimationClip'у, которые будут делать то же что делают originalKeyframes'ы.
    //     * Новые кифреймы должны оперировать на новых геймобжах. Вообще не обязательно их щас делать. Нам сейчас нужны только ссылки на спрайты.
    //     * Геймобжы и целевые спрайтрендереры можно пока не создавать - достаточно прописать пути для них, а создать потом. В дальнейшем этот 
    //     * AnimationClip будет привязан к параметру Motion какого-нибудь State'а StateMachine'ы какого-нибудь Animator'а, являющегося оптимизированным
    //     * двойником какого-нибудь оригинального исходного Animator'а в сцене или в ассетах. Потом, при создании оптимизированных копий геймобъектов в  
    //     * сцене или в ресурсах можно будет воссоздать нужную структуру спрайтрендереров.
    //     */

    //    var rootPath = originalBinding.path;

    //    var optimizedKeyframes = new ObjectReferenceKeyframe();

    //    for (int i = 0; i < originalKeyframes.Length; i++)
    //    {
    //        if (!(originalKeyframes[i].value is Sprite))
    //            continue;
    //        var spriteReference = originalKeyframes[i].value as Sprite;
    //        var time = originalKeyframes[i].time;

    //        var texture = spriteReference.texture;
    //        var mask = new bool[texture.width, texture.height];

    //        var areaHeight = 4;
    //        var areaWidth = 4;

    //        for (int x = 0; x < texture.width; x++)
    //        {
    //            for (int y = 0; y < texture.height; y++)
    //            {
    //                var color = texture.GetPixel(x, y);
    //                if (color.a == 0f)
    //                    continue;

    //                if (mask[x, y] == true)
    //                    continue;

    //                var allowedWidth = 0;
    //                var allowedHeight = 0;
    //                for (int xx = 0; xx < areaWidth; xx++)
    //                {
    //                    for (int yy = 0; yy < areaHeight; yy++)
    //                    {
    //                        if (mask[x + xx, y + yy])
    //                        {
    //                            resolutionX = xx;
    //                            break;
    //                        }
    //                        allowedWidth = xx + 1;
    //                        allowedHeight = yy + 1;
    //                    }
    //                }
    //                if (mask[x + 1, y + 1])
    //                {
    //                    x--;
    //                    y--;
    //                }
    //                else if (mask[x, y + 1])
    //                    y--;
    //                else if (mask[x + 1, y])
    //                    x--;

    //                var colors = texture.GetPixels(x, y, 2, 2);
    //            }
    //        }
    //    }
    //}

    //private static ObjectReferenceKeyframe[] getOptimizedObjectReferenceKeyframes(ObjectReferenceKeyframe[] originalKeyframes)
    //{
    //    /*
    //     * Ок. Вот тут собственно и нужно проделывать все манипуляции, ибо тут мы имеем кадр и время, в которое его надо показывать
    //     */

    //    var optimizedKeyframes = new ObjectReferenceKeyframe[originalKeyframes.Length];
    //    for (int i = 0; i < originalKeyframes.Length; i++)
    //    {
    //        var originalKeyframe = originalKeyframes[i];

    //        var optimizedKeyframe = new ObjectReferenceKeyframe();
    //        optimizedKeyframe.value = originalKeyframe.value;
    //        optimizedKeyframe.time = originalKeyframe.time;
    //        optimizedKeyframes[i] = optimizedKeyframe;
    //    }

    //    return optimizedKeyframes;
    //}

    private static AnimationEvent[] getOptimizedAnimationClipEvents(AnimationEvent[] events)
    {
        var result = new AnimationEvent[events.Length];

        for (int i = 0; i < events.Length; i++)
        {
            result[i] = new AnimationEvent();
            result[i].data = events[i].data;
            result[i].floatParameter = events[i].floatParameter;
            result[i].functionName = events[i].functionName;
            result[i].intParameter = events[i].intParameter;
            result[i].messageOptions = events[i].messageOptions;
            result[i].objectReferenceParameter = events[i].objectReferenceParameter;
            result[i].stringParameter = events[i].stringParameter;
            result[i].time = events[i].time;
        }

        return result;
    }

    private static AnimatorStateTransition[] getOptimizedAnimatorStateTransition(AnimatorStateTransition[] originalTransitions, GameObject prefab, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optTransitions = new AnimatorStateTransition[originalTransitions.Length];

        for (int i = 0; i < originalTransitions.Length; i++)
        {
            var originalTransition = originalTransitions[i];
            var newTransition = new AnimatorStateTransition();

            newTransition.canTransitionToSelf = originalTransition.canTransitionToSelf;
            var optConditions = new AnimatorCondition[originalTransition.conditions.Length];
            for (int j = 0; j < originalTransition.conditions.Length; j++)
            {
                var newCondition = new AnimatorCondition();
                var originalCondition = originalTransition.conditions[j];

                newCondition.mode = originalCondition.mode;
                newCondition.parameter = originalCondition.parameter;
                newCondition.threshold = originalCondition.threshold;

                optConditions[j] = newCondition;
            }
            newTransition.conditions = optConditions;
            newTransition.destinationState = getOptReference(originalTransition.destinationState, prefab, originalToOptObjectReferences, animationClipsFolder);
            newTransition.destinationStateMachine = getOptReference(originalTransition.destinationStateMachine, prefab, originalToOptObjectReferences, animationClipsFolder);
            newTransition.exitTime = originalTransition.exitTime;
            newTransition.hasExitTime = originalTransition.hasExitTime;
            newTransition.hasFixedDuration = originalTransition.hasFixedDuration;
            newTransition.hideFlags = originalTransition.hideFlags;
            newTransition.interruptionSource = originalTransition.interruptionSource;
            newTransition.isExit = originalTransition.isExit;
            newTransition.mute = originalTransition.mute;
            newTransition.name = originalTransition.name;
            newTransition.offset = originalTransition.offset;
            newTransition.orderedInterruption = originalTransition.orderedInterruption;
            newTransition.solo = originalTransition.solo;

            optTransitions[i] = newTransition;
        }

        return optTransitions;
    }
}
