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

    internal static void Do(AnimatorController originalCtrlr, OptimizedControllerStructure structure, string folderPath, params UnityOptimizedSpritesStructure[] chunksInfo)
    {
        _chunksInfos = chunksInfo;

        var path = Path.Combine(folderPath, $"{originalCtrlr.name}-optimized.controller");
        Debug.Log($"Do {path}");
        var optCtrlr = AnimatorController.CreateAnimatorControllerAtPath(path);
        optCtrlr.RemoveLayer(0);

        var animationClipsFolder = Path.Combine(folderPath, "AnimationClips");
        if (!Directory.Exists(animationClipsFolder))
            Directory.CreateDirectory(animationClipsFolder);

        var originalToOptObjectReferences = new Dictionary<UnityEngine.Object, UnityEngine.Object>();

        for (int i = 0; i < originalCtrlr.layers.Length; i++)
        {
            var optLayer = getOptimizedLayer(originalCtrlr.layers[i], structure, originalToOptObjectReferences, animationClipsFolder);
            optCtrlr.AddLayer(optLayer);
        }
    }

    private static T getOptReference<T>(T original, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder) where T : UnityEngine.Object
    {
        if (original == null)
            return null;
        if (!originalToOptObjectReferences.ContainsKey(original))
        {
            if (typeof(T).Equals(typeof(AnimatorStateMachine)))
                return (T)(UnityEngine.Object)getOptimizedStateMachine(original as AnimatorStateMachine, structure, originalToOptObjectReferences, animationClipsFolder);
            else if (typeof(T).Equals(typeof(AnimatorState)))
                return (T)(UnityEngine.Object)getOptimizedState(original as AnimatorState, structure, originalToOptObjectReferences, animationClipsFolder);
            else
                throw new ApplicationException($"Unknown reference type occured :{typeof(T).FullName}!");
        }
        else
            return (T)originalToOptObjectReferences[original];
    }

    private static AnimatorControllerLayer getOptimizedLayer(AnimatorControllerLayer originalLayer, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optLayer = new AnimatorControllerLayer();

        optLayer.blendingMode = originalLayer.blendingMode;
        optLayer.defaultWeight = originalLayer.defaultWeight;
        optLayer.iKPass = originalLayer.iKPass;
        optLayer.name = originalLayer.name;
        optLayer.stateMachine = getOptReference(originalLayer.stateMachine, structure, originalToOptObjectReferences, animationClipsFolder);
        optLayer.syncedLayerAffectsTiming = originalLayer.syncedLayerAffectsTiming;
        optLayer.syncedLayerIndex = originalLayer.syncedLayerIndex;

        return optLayer;
    }

    private static AnimatorStateMachine getOptimizedStateMachine(AnimatorStateMachine originalStateMachine, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optStateMachine = new AnimatorStateMachine();
        originalToOptObjectReferences.Add(originalStateMachine, optStateMachine);

        optStateMachine.anyStatePosition = originalStateMachine.anyStatePosition;
        optStateMachine.entryPosition = originalStateMachine.entryPosition;
        optStateMachine.exitPosition = originalStateMachine.exitPosition;
        optStateMachine.name = originalStateMachine.name;
        optStateMachine.parentStateMachinePosition = originalStateMachine.parentStateMachinePosition;

        for (int i = 0; i < originalStateMachine.states.Length; i++)
            optStateMachine.AddState(getOptReference(originalStateMachine.states[i].state, structure, originalToOptObjectReferences, animationClipsFolder), originalStateMachine.states[i].position);

        //for (int i = 0; i < originalStateMachine.anyStateTransitions.Length; i++)
        //{
        //    originalStateMachine.AddAnyStateTransition
        //}

        return optStateMachine;
    }

    private static AnimatorState getOptimizedState(AnimatorState originalAnimatorState, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
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
        optAnimatorState.motion = getOptimizedMotion(originalAnimatorState.motion, structure, originalToOptObjectReferences, animationClipsFolder);
        optAnimatorState.name = originalAnimatorState.name;
        optAnimatorState.speed = originalAnimatorState.speed;
        optAnimatorState.speedParameter = originalAnimatorState.speedParameter;
        optAnimatorState.speedParameterActive = originalAnimatorState.speedParameterActive;
        optAnimatorState.tag = originalAnimatorState.tag;
        optAnimatorState.timeParameter = originalAnimatorState.timeParameter;
        optAnimatorState.timeParameterActive = originalAnimatorState.timeParameterActive;
        optAnimatorState.writeDefaultValues = originalAnimatorState.writeDefaultValues;

        optAnimatorState.transitions = getOptimizedAnimatorStateTransition(originalAnimatorState.transitions, structure, originalToOptObjectReferences, animationClipsFolder);

        return optAnimatorState;
    }

    private static Motion getOptimizedMotion(Motion originalMotion, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        if (!(originalMotion is AnimationClip))
            throw new ApplicationException($"Unknown type of motion - {originalMotion.GetType().FullName}. Never done this before...");

        var originalAnimationClip = originalMotion as AnimationClip;

        var optMotion = new AnimationClip();
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
                var editorCurve = AnimationUtility.GetEditorCurve(originalMotion as AnimationClip, bindings[i]);
            }
        //var optimizedBindingsList = new List<EditorCurveBinding>();
        for (int i = 0; i < objectReferenceBindings.Length; i++)
        {
            var originalBinding = objectReferenceBindings[i];
            Debug.Log($"originalBinding.propertyName = {originalBinding.propertyName}");
            Debug.Log($"originalBinding.path = {originalBinding.path}");
            Debug.Log($"originalBinding.type = {originalBinding.type.FullName}");
            var keyframes = AnimationUtility.GetObjectReferenceCurve(originalAnimationClip, originalBinding);
            setOptBindings(optMotion, structure, originalBinding, keyframes);
            //setOptimizedCurvesBasedOnOriginalCurve(optMotion, structure, originalBinding, keyframes);
            //AnimationUtility.SetObjectReferenceCurve(optMotion, originalBinding, optimizedBindingsList.ToArray());
        }

        var animationClipPath = Path.Combine(animationClipsFolder, $"{originalAnimationClip.name}.anim");
        AssetDatabase.CreateAsset(optMotion, animationClipPath);
        return optMotion;
    }

    private class BindingKeyframes
    {
        public EditorCurveBinding SpriteBinding;
        public EditorCurveBinding TransformBindingX;
        public EditorCurveBinding TransformBindingY;
        public ObjectReferenceKeyframe[] SpriteKeyframes;
        public AnimationCurve TransformCurveX;
        public AnimationCurve TransformCurveY;
    }

    private static void setOptBindings(AnimationClip optMotion, OptimizedControllerStructure structure, EditorCurveBinding originalBinding, ObjectReferenceKeyframe[] keyframes)
    {
        /*
         * Ок, тут мы имеем на входе 1 дорожку - originalBinding и кифреймы, описанные keyframes. И задача у нас - 
         * проставить optMotion набор дорожек, заменяющих входящую дорожку.
         */


        //Сначала нам надо понять сколько дорожек нам понадобится.
        var maxOptSpritesCount = 0;
        for (int i = 0; i < keyframes.Length; i++)
        {
            var currentChunksCount = 0;
            var keyframe = keyframes[i];
            if (keyframe.value == null)
                continue;
            if (!(keyframe.value is Sprite))
                throw new ApplicationException($"keyframe.value is not Sprite. Don't know what to do.");

            var chunksInfo = _chunksInfos.Where(info => info.Sprites.Contains(keyframe.value));
            if (!chunksInfo.Any()) //Если оптимизированной структуры не найдено, то просто ставим оригинал, т.е. 1 спрайт.
            {
                currentChunksCount = 1;
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
            currentChunksCount = chunks.Array.Length;

            if (currentChunksCount > maxOptSpritesCount)
                maxOptSpritesCount = currentChunksCount;
        }
        Debug.Log($"Ок, максимальное кол-во дорожек у нас {maxOptSpritesCount}");

        var optBindings = new List<BindingKeyframes>();
        for (int i = 0; i < maxOptSpritesCount; i++)
        {
            var binds = new BindingKeyframes();
            var spriteBinding = new EditorCurveBinding();
            var transformBindingX = new EditorCurveBinding();
            var transformBindingY = new EditorCurveBinding();

            spriteBinding.propertyName = "m_Sprite";
            spriteBinding.path = $"{originalBinding.path}/OptimizerSpriteRenderer_{i}";
            spriteBinding.type = typeof(SpriteRenderer);

            transformBindingX.propertyName = "m_LocalPosition.x";
            transformBindingX.path = $"{originalBinding.path}/OptimizerSpriteRenderer_{i}";
            transformBindingX.type = typeof(Transform);

            transformBindingY.propertyName = "m_LocalPosition.y";
            transformBindingY.path = $"{originalBinding.path}/OptimizerSpriteRenderer_{i}";
            transformBindingY.type = typeof(Transform);

            binds.SpriteBinding = spriteBinding;
            binds.TransformBindingX = transformBindingX;
            binds.TransformBindingY = transformBindingY;

            binds.SpriteKeyframes = new ObjectReferenceKeyframe[keyframes.Length];
            binds.TransformCurveX = new AnimationCurve();
            binds.TransformCurveX.keys = new Keyframe[keyframes.Length];
            binds.TransformCurveY = new AnimationCurve();
            binds.TransformCurveY.keys = new Keyframe[keyframes.Length];
            optBindings.Add(binds);
        }

        for (int i = 0; i < keyframes.Length; i++)
        {
            var keyframe = keyframes[i];
            if (keyframe.value == default)
            {
                for (int j = 0; j < optBindings.Count; j++)
                {
                    optBindings[j].SpriteKeyframes[i] = new ObjectReferenceKeyframe();
                    optBindings[j].SpriteKeyframes[i].time = keyframe.time;
                    optBindings[j].SpriteKeyframes[i].value = default;
                }

                continue;
            }
            if (!(keyframe.value is Sprite))
                throw new ApplicationException($"keyframe.value is not Sprite. Don't know what to do.");

            var chunksInfo = _chunksInfos.Where(info => info.Sprites.Contains(keyframe.value));
            if (!chunksInfo.Any()) //Если оптимизированной структуры не найдено, то просто ставим оригинал
            {
                optBindings[0].SpriteKeyframes[i] = new ObjectReferenceKeyframe();
                optBindings[0].SpriteKeyframes[i].time = keyframe.time;
                optBindings[0].SpriteKeyframes[i].value = keyframe.value;

                optBindings[0].TransformCurveX.keys[i] = new Keyframe();
                optBindings[0].TransformCurveX.keys[i].time = keyframe.time;
                optBindings[0].TransformCurveX.keys[i].value = 0;

                optBindings[0].TransformCurveY.keys[i] = new Keyframe();
                optBindings[0].TransformCurveY.keys[i].time = keyframe.time;
                optBindings[0].TransformCurveY.keys[i].value = 0;

                for (int j = 1; j < optBindings.Count; j++)
                {
                    optBindings[j].SpriteKeyframes[i] = new ObjectReferenceKeyframe();
                    optBindings[j].SpriteKeyframes[i].time = keyframe.time;
                    optBindings[j].SpriteKeyframes[i].value = default;
                }
                continue;
            }
            
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

            for (int j = 0; j < optBindings.Count; j++)
            {
                optBindings[j].SpriteKeyframes[i] = new ObjectReferenceKeyframe();
                optBindings[j].SpriteKeyframes[i].time = keyframe.time;
                optBindings[j].SpriteKeyframes[i].value = chunks.Array.;
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

    private static AnimatorStateTransition[] getOptimizedAnimatorStateTransition(AnimatorStateTransition[] originalTransitions, OptimizedControllerStructure structure, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
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
            newTransition.destinationState = getOptReference(originalTransition.destinationState, structure, originalToOptObjectReferences, animationClipsFolder);
            newTransition.destinationStateMachine = getOptReference(originalTransition.destinationStateMachine, structure, originalToOptObjectReferences, animationClipsFolder);
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
