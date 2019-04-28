using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using static SpritesheetOptimizerWindow;

public static class AnimatorControllerDoer
{
    internal static void Do(AnimatorController originalCtrlr, OptimizedControllerStructure structure, List<SpritesheetChunk> futureSpriteSheet, string folderPath)
    {
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
            var optLayer = getOptimizedLayer(originalCtrlr.layers[i], originalToOptObjectReferences, animationClipsFolder);
            optCtrlr.AddLayer(optLayer);
        }
    }

    private static T getOptReference<T>(T original, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder) where T : UnityEngine.Object
    {
        if (original == null)
            return null;
        if (!originalToOptObjectReferences.ContainsKey(original))
        {
            if (typeof(T).Equals(typeof(AnimatorStateMachine)))
                return (T)(UnityEngine.Object)getOptimizedStateMachine(original as AnimatorStateMachine, originalToOptObjectReferences, animationClipsFolder);
            else if (typeof(T).Equals(typeof(AnimatorState)))
                return (T)(UnityEngine.Object)getOptimizedState(original as AnimatorState, originalToOptObjectReferences, animationClipsFolder);
            else
                throw new ApplicationException($"Unknown reference type occured :{typeof(T).FullName}!");
        }
        else
            return (T)originalToOptObjectReferences[original];
    }

    private static AnimatorControllerLayer getOptimizedLayer(AnimatorControllerLayer originalLayer, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optLayer = new AnimatorControllerLayer();

        optLayer.blendingMode = originalLayer.blendingMode;
        optLayer.defaultWeight = originalLayer.defaultWeight;
        optLayer.iKPass = originalLayer.iKPass;
        optLayer.name = originalLayer.name;
        optLayer.stateMachine = getOptReference(originalLayer.stateMachine, originalToOptObjectReferences, animationClipsFolder);
        optLayer.syncedLayerAffectsTiming = originalLayer.syncedLayerAffectsTiming;
        optLayer.syncedLayerIndex = originalLayer.syncedLayerIndex;

        return optLayer;
    }

    private static AnimatorStateMachine getOptimizedStateMachine(AnimatorStateMachine originalStateMachine, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
    {
        var optStateMachine = new AnimatorStateMachine();
        originalToOptObjectReferences.Add(originalStateMachine, optStateMachine);

        optStateMachine.anyStatePosition = originalStateMachine.anyStatePosition;
        optStateMachine.entryPosition = originalStateMachine.entryPosition;
        optStateMachine.exitPosition = originalStateMachine.exitPosition;
        optStateMachine.name = originalStateMachine.name;
        optStateMachine.parentStateMachinePosition = originalStateMachine.parentStateMachinePosition;

        for (int i = 0; i < originalStateMachine.states.Length; i++)
            optStateMachine.AddState(getOptReference(originalStateMachine.states[i].state, originalToOptObjectReferences, animationClipsFolder), originalStateMachine.states[i].position);

        //for (int i = 0; i < originalStateMachine.anyStateTransitions.Length; i++)
        //{
        //    originalStateMachine.AddAnyStateTransition
        //}

        return optStateMachine;
    }

    private static AnimatorState getOptimizedState(AnimatorState originalAnimatorState, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
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
        optAnimatorState.motion = getOptimizedMotion(originalAnimatorState.motion, originalToOptObjectReferences, animationClipsFolder);
        optAnimatorState.name = originalAnimatorState.name;
        optAnimatorState.speed = originalAnimatorState.speed;
        optAnimatorState.speedParameter = originalAnimatorState.speedParameter;
        optAnimatorState.speedParameterActive = originalAnimatorState.speedParameterActive;
        optAnimatorState.tag = originalAnimatorState.tag;
        optAnimatorState.timeParameter = originalAnimatorState.timeParameter;
        optAnimatorState.timeParameterActive = originalAnimatorState.timeParameterActive;
        optAnimatorState.writeDefaultValues = originalAnimatorState.writeDefaultValues;

        optAnimatorState.transitions = getOptimizedAnimatorStateTransition(originalAnimatorState.transitions, originalToOptObjectReferences, animationClipsFolder);

        return optAnimatorState;
    }

    private static Motion getOptimizedMotion(Motion originalMotion, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
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

        var objectReferenceBindings = AnimationUtility.GetObjectReferenceCurveBindings(originalAnimationClip);
        for (int i = 0; i < objectReferenceBindings.Length; i++)
        {
            var originalBinding = objectReferenceBindings[i];
            Debug.Log($"originalBinding.path = {originalBinding.path}");
            var optimizedBinding = new EditorCurveBinding();
            var keyframes = AnimationUtility.GetObjectReferenceCurve(originalAnimationClip, originalBinding);
            AnimationUtility.SetObjectReferenceCurve(optMotion, originalBinding, getOptimizedObjectReferenceKeyframes(keyframes));
        }

        var animationClipPath = Path.Combine(animationClipsFolder, $"{originalAnimationClip.name}.anim");
        AssetDatabase.CreateAsset(optMotion, animationClipPath);
        return optMotion;
    }

    private static ObjectReferenceKeyframe[] getOptimizedObjectReferenceKeyframes(ObjectReferenceKeyframe[] originalKeyframes)
    {
        var optimizedKeyframes = new ObjectReferenceKeyframe[originalKeyframes.Length];
        for (int i = 0; i < originalKeyframes.Length; i++)
        {
            var originalKeyframe = originalKeyframes[i];

            var optimizedKeyframe = new ObjectReferenceKeyframe();
            optimizedKeyframe.value = originalKeyframe.value;
            optimizedKeyframe.time = originalKeyframe.time;
            optimizedKeyframes[i] = optimizedKeyframe;
        }

        return optimizedKeyframes;
    }

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

    private static AnimatorStateTransition[] getOptimizedAnimatorStateTransition(AnimatorStateTransition[] originalTransitions, Dictionary<UnityEngine.Object, UnityEngine.Object> originalToOptObjectReferences, string animationClipsFolder)
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
            newTransition.destinationState = getOptReference(originalTransition.destinationState, originalToOptObjectReferences, animationClipsFolder);
            newTransition.destinationStateMachine = getOptReference(originalTransition.destinationStateMachine, originalToOptObjectReferences, animationClipsFolder);
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
